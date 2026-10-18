"""CardioAgent: MLP risk models + medical knowledge base + uniform-cost search for lifestyle plans."""
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier

from cardio_utils import RANDOM_STATE, load_clean_data, split_and_scale

RESULTS = Path(__file__).parent / "results"

VALID_RANGES = {
    "age_years": (18, 100), "height_cm": (130, 210), "weight_kg": (35, 200),
    "ap_hi": (80, 250), "ap_lo": (40, 160),
}
CATEGORIES = {"cholesterol": {1, 2, 3}, "gluc": {1, 2, 3}, "smoke": {0, 1}, "alco": {0, 1},
              "active": {0, 1}, "is_male": {0, 1}}
TRAINING_AGE_RANGE = (29, 65)

QUICK_INPUTS = ["age_years", "ap_hi", "cholesterol"]
FULL_INPUTS = ["age_years", "is_male", "height_cm", "weight_kg", "ap_hi", "ap_lo",
               "cholesterol", "gluc", "smoke", "alco", "active"]

MEASURABLE_ACTIONS = {
    "lower_bp":    {"cost": 2, "label": "Lower blood pressure by 10/5 mmHg (toward 120/80)"},
    "lower_chol":  {"cost": 2, "label": "Lower cholesterol by one level"},
    "lose_weight": {"cost": 2, "label": "Lose weight (BMI −2, toward 25)"},
    "be_active":   {"cost": 1, "label": "Become physically active"},
}
# Medically recommended, but the model wrongly predicts higher risk after them: advised from rules only.
GUIDELINE_ONLY = {
    "smoke": "Stop smoking",
    "alco": "Reduce or stop alcohol",
    "gluc": "Bring blood sugar (glucose) to normal",
}
DISCLAIMER = ("This is a decision-support estimate from a model trained on 70,000 checkups. "
              "It is not a diagnosis. Discuss any changes with a doctor.")


def choose_threshold(y_true, proba, target_recall=0.80):
    """Highest threshold (steps of 0.01) that still catches `target_recall` of sick patients."""
    for t in np.arange(0.90, 0.05, -0.01):
        if recall_score(y_true, (proba >= t).astype(int)) >= target_recall:
            return round(float(t), 2)
    return 0.05


class CardioAgent:
    """Perceive (validate input) -> think (predict, profile, plan) -> act (report)."""

    def __init__(self):
        data = split_and_scale(load_clean_data())
        self.features = data["features"]
        self.scaler = data["scaler"]

        self.full_model = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                                        early_stopping=True, max_iter=300, random_state=RANDOM_STATE)
        self.full_model.fit(data["X_train_s"], data["y_train"])
        self.full_threshold = choose_threshold(
            data["y_val"], self.full_model.predict_proba(data["X_val_s"])[:, 1])

        ga = json.loads((RESULTS / "ga_best_design.json").read_text())
        self.quick_features = ga["features"]
        self.quick_cols = [self.features.index(f) for f in self.quick_features]
        self.quick_model = MLPClassifier(hidden_layer_sizes=tuple(ga["shape"]), activation="relu",
                                         early_stopping=True, max_iter=200, random_state=RANDOM_STATE)
        self.quick_model.fit(data["X_train_s"][:, self.quick_cols], data["y_train"])
        self.quick_threshold = choose_threshold(
            data["y_val"], self.quick_model.predict_proba(data["X_val_s"][:, self.quick_cols])[:, 1])

        prof = json.loads((RESULTS / "patient_profiles.json").read_text())
        self.centres = np.array(prof["centres_scaled"])
        self.groups = {g["cluster"]: g for g in prof["groups"]}
        self.baseline = min(prof["groups"], key=lambda g: g["disease_rate"])

        self.medians = data["X_train"].median()

    def perceive(self, patient):
        """Check the input. Returns (mode, errors, warnings)."""
        errors, warnings = [], []
        given = {k for k, v in patient.items() if v is not None}

        if set(FULL_INPUTS) <= given:
            mode = "full"
        elif set(QUICK_INPUTS) <= given:
            mode = "quick"
            warnings.append("Only age, blood pressure and cholesterol given: running a QUICK screening. "
                            "Add the other values for a full assessment with advice.")
        else:
            missing = sorted(set(QUICK_INPUTS) - given)
            return None, [f"Missing required values: {missing}"], warnings

        for key, (lo, hi) in VALID_RANGES.items():
            if key in given and not lo <= patient[key] <= hi:
                errors.append(f"{key} = {patient[key]} is outside the valid range {lo}–{hi}")
        for key, allowed in CATEGORIES.items():
            if key in given and patient[key] not in allowed:
                errors.append(f"{key} = {patient[key]} must be one of {sorted(allowed)}")
        if "ap_lo" in given and patient["ap_hi"] <= patient["ap_lo"]:
            errors.append("Systolic blood pressure (ap_hi) must be higher than diastolic (ap_lo)")
        if not TRAINING_AGE_RANGE[0] <= patient["age_years"] <= TRAINING_AGE_RANGE[1]:
            warnings.append(f"Age {patient['age_years']} is outside the ages in the training data "
                            f"({TRAINING_AGE_RANGE[0]}–{TRAINING_AGE_RANGE[1]}); the estimate is less reliable.")
        return mode, errors, warnings

    def _row(self, p):
        """Patient dict -> one row of the 10 model features (raw units)."""
        return {
            "ap_hi": p["ap_hi"], "ap_lo": p.get("ap_lo"), "cholesterol": p["cholesterol"],
            "gluc": p.get("gluc"), "smoke": p.get("smoke"), "alco": p.get("alco"),
            "active": p.get("active"), "age_years": p["age_years"],
            "bmi": p["weight_kg"] / (p["height_cm"] / 100) ** 2 if p.get("weight_kg") else None,
            "is_male": p.get("is_male"),
        }

    def _scaled(self, rows):
        df = pd.DataFrame(rows)[self.features].astype(float)
        df = df.fillna(self.medians)
        return self.scaler.transform(df)

    def full_risk(self, rows):
        return self.full_model.predict_proba(self._scaled(rows))[:, 1]

    def quick_risk(self, rows):
        return self.quick_model.predict_proba(self._scaled(rows)[:, self.quick_cols])[:, 1]

    @staticmethod
    def risk_level(risk, threshold):
        if risk < threshold:
            return "LOW"
        return "HIGH" if risk >= 0.70 else "ELEVATED"

    def profile(self, row):
        """Nearest k-means centre (Part 3)."""
        x = self._scaled([row])[0]
        return self.groups[int(np.argmin(((self.centres - x) ** 2).sum(axis=1)))]

    @staticmethod
    def _apply(row, height_cm, state):
        """Apply a plan to a patient row. state = (bp_steps, chol_steps, weight_steps, active)."""
        bp, chol, weight, active = state
        r = dict(row)
        if bp:
            r["ap_hi"] = max(row["ap_hi"] - 10 * bp, min(row["ap_hi"], 120))
            r["ap_lo"] = max(row["ap_lo"] - 5 * bp, min(row["ap_lo"], 80))
            r["ap_lo"] = min(r["ap_lo"], r["ap_hi"] - 10)
        if chol:
            r["cholesterol"] = row["cholesterol"] - chol
        if weight:
            r["bmi"] = max(row["bmi"] - 2 * weight, min(row["bmi"], 25.0))
        if active:
            r["active"] = 1
        return r

    @staticmethod
    def _max_steps(row):
        """How many steps of each change make sense for this patient (0 = not applicable)."""
        return (
            max(0, int(np.ceil((row["ap_hi"] - 120) / 10))),
            row["cholesterol"] - 1,
            max(0, int(np.ceil((row["bmi"] - 25) / 2))),
            0 if row["active"] == 1 else 1,
        )

    @staticmethod
    def describe_changes(row, new, height_cm):
        """Plain-language description of the difference between two patient rows."""
        out = []
        if new["ap_hi"] < row["ap_hi"]:
            out.append(f"Lower blood pressure from {row['ap_hi']:.0f}/{row['ap_lo']:.0f} "
                       f"to {new['ap_hi']:.0f}/{new['ap_lo']:.0f}")
        if new["cholesterol"] < row["cholesterol"]:
            out.append(f"Lower cholesterol from level {row['cholesterol']} to {new['cholesterol']}"
                       + (" (normal)" if new["cholesterol"] == 1 else ""))
        if new["bmi"] < row["bmi"]:
            kg = (row["bmi"] - new["bmi"]) * (height_cm / 100) ** 2
            out.append(f"Lose about {kg:.0f} kg (BMI {row['bmi']:.1f} → {new['bmi']:.1f})")
        if new["active"] > row["active"]:
            out.append("Become physically active")
        return out

    def plan(self, row, height_cm):
        """Uniform-cost search over (bp, chol, weight, active) steps for the cheapest plan with risk < threshold."""
        max_steps = self._max_steps(row)
        action_index = {"lower_bp": 0, "lower_chol": 1, "lose_weight": 2, "be_active": 3}

        risk_cache = {}
        def risk_of(state):
            if state not in risk_cache:
                risk_cache[state] = float(self.full_risk([self._apply(row, height_cm, state)])[0])
            return risk_cache[state]

        start = (0, 0, 0, 0)
        frontier = [(0, risk_of(start), start, [])]
        best_seen = (risk_of(start), 0, start, [])
        visited = set()

        while frontier:
            cost, risk, state, actions = heapq.heappop(frontier)
            if state in visited:
                continue
            visited.add(state)
            if risk < best_seen[0]:
                best_seen = (risk, cost, state, actions)
            if risk < self.full_threshold:
                return {"reached_goal": True, "cost": cost, "risk": risk,
                        "actions": actions, "state": state, "states_explored": len(visited)}
            for name, i in action_index.items():
                if state[i] < max_steps[i]:
                    nxt = list(state); nxt[i] += 1; nxt = tuple(nxt)
                    if nxt not in visited:
                        heapq.heappush(frontier, (cost + MEASURABLE_ACTIONS[name]["cost"],
                                                  risk_of(nxt), nxt, actions + [name]))

        risk, cost, state, actions = best_seen
        return {"reached_goal": False, "cost": cost, "risk": risk,
                "actions": actions, "state": state, "states_explored": len(visited)}

    def assess(self, patient):
        """Run the full perceive -> think -> act cycle. Returns a dict (see report())."""
        mode, errors, warnings = self.perceive(patient)
        result = {"mode": mode, "errors": errors, "warnings": warnings}
        if errors:
            return result

        row = self._row(patient)
        if mode == "quick":
            risk = float(self.quick_risk([row])[0])
            result.update(risk=risk, threshold=self.quick_threshold,
                          level=self.risk_level(risk, self.quick_threshold))
            return result

        risk = float(self.full_risk([row])[0])
        group = self.profile(row)
        result.update(risk=risk, threshold=self.full_threshold,
                      level=self.risk_level(risk, self.full_threshold),
                      profile=group, bmi=row["bmi"])

        base = self.baseline["average_patient"]
        factors = []
        if row["ap_hi"] >= 130:
            factors.append(f"blood pressure {patient['ap_hi']}/{patient['ap_lo']} "
                           f"(low-risk group average {base['ap_hi']:.0f}/{base['ap_lo']:.0f})")
        if row["cholesterol"] > 1:
            factors.append(f"cholesterol level {patient['cholesterol']} (normal = 1)")
        if row["bmi"] >= 30:
            factors.append(f"BMI {row['bmi']:.1f} (obese range ≥ 30)")
        elif row["bmi"] >= 25:
            factors.append(f"BMI {row['bmi']:.1f} (overweight range 25–30)")
        if row["age_years"] >= 55:
            factors.append(f"age {patient['age_years']} (risk rises with age; not changeable)")
        result["factors"] = factors

        single = []
        max_steps = self._max_steps(row)
        for i in range(4):
            if max_steps[i] == 0:
                continue
            state = [0, 0, 0, 0]; state[i] = max_steps[i]
            changed = self._apply(row, patient["height_cm"], tuple(state))
            new_risk = float(self.full_risk([changed])[0])
            corrected = new_risk > risk
            new_risk = min(new_risk, risk)
            single.append((self.describe_changes(row, changed, patient["height_cm"])[0], new_risk, corrected))
        result["what_if"] = sorted(single, key=lambda item: item[1])

        result["plan"] = self.plan(row, patient["height_cm"])
        final = self._apply(row, patient["height_cm"], result["plan"]["state"])
        result["plan"]["steps"] = self.describe_changes(row, final, patient["height_cm"])
        result["guideline_advice"] = [msg for key, msg in GUIDELINE_ONLY.items()
                                      if (patient[key] == 1 if key != "gluc" else patient[key] > 1)]
        return result

    def report(self, patient):
        """Human-readable report."""
        r = self.assess(patient)
        lines = ["=" * 66, "CARDIOAGENT REPORT", "=" * 66]
        if r["errors"]:
            lines += ["Cannot assess this patient:"] + [f"  ✘ {e}" for e in r["errors"]]
            return "\n".join(lines)
        lines += [f"  ⚠ {w}" for w in r["warnings"]]
        lines.append(f"Mode: {r['mode'].upper()}   |   Estimated risk: {r['risk']:.0%}   |   "
                     f"Level: {r['level']}  (follow-up threshold {r['threshold']:.0%})")
        if r["mode"] == "quick":
            lines += ["", DISCLAIMER]
            return "\n".join(lines)

        g = r["profile"]
        lines += ["", f"Patient profile: '{g['name']}' — in this group {g['disease_rate']:.0%} "
                      f"of patients have heart disease."]
        if abs(r["risk"] - g["disease_rate"]) >= 0.20:
            direction = "higher" if r["risk"] > g["disease_rate"] else "lower"
            reason = " because of your individual values below." if r["factors"] else "."
            lines.append(f"  Your personal estimate is {direction} than your group's average{reason}")
        if r["factors"]:
            lines += ["", "Main factors:"] + [f"  • {f}" for f in r["factors"]]
        if r["what_if"]:
            lines += ["", "What-if (one change at a time):"]
            for text, new, corrected in r["what_if"]:
                note = "   (no measurable benefit in the model)" if corrected else ""
                lines.append(f"  {text:<52} {r['risk']:.0%} → {new:.0%}{note}")

        p = r["plan"]
        lines.append("")
        if r["level"] == "LOW":
            lines.append("Risk is below the follow-up threshold. Keep up healthy habits.")
        elif not p["actions"]:
            lines.append("No measurable lifestyle change is available in the model for this patient.")
        else:
            head = ("Suggested plan (easiest set of changes that brings risk below the threshold):"
                    if p["reached_goal"] else
                    "No plan brings risk below the threshold (age and other fixed factors weigh in). "
                    "Plan with the largest reduction:")
            lines.append(head)
            lines += [f"  ✔ {step}" for step in p["steps"]]
            lines.append(f"  → estimated risk {r['risk']:.0%} → {p['risk']:.0%}   "
                         f"(effort {p['cost']}, {p['states_explored']} plans searched)")
        if r["guideline_advice"]:
            lines += ["", "Also recommended by medical guidelines (effect not measurable in this dataset):"]
            lines += [f"  ✔ {a}" for a in r["guideline_advice"]]
        lines += ["", DISCLAIMER]
        return "\n".join(lines)
