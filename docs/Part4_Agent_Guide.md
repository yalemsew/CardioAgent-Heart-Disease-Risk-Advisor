# Part 4: CardioAgent, the Intelligent Agent

**Project:** CardioAgent, heart disease risk prediction (CS582 Machine Learning)
**Notebook:** [`04_agent.ipynb`](../04_agent.ipynb) · **Agent code:** [`cardio_agent.py`](../cardio_agent.py) · **Runtime:** ~15 seconds
**Course link:** Lecture 11 (agents, PEAS), Lecture 12 (search), plus everything from Parts 1–3

---

## Contents

1. [In simple words (read this first)](#1-in-simple-words-read-this-first)
2. [Files](#2-files)
3. [What is an agent? PEAS](#3-what-is-an-agent-peas)
4. [How the agent is built from Parts 1–3](#4-how-the-agent-is-built-from-parts-13)
5. [The safety check: a surprising finding](#5-the-safety-check-a-surprising-finding)
6. [The knowledge base](#6-the-knowledge-base)
7. [Perceive: checking the input](#7-perceive-checking-the-input)
8. [Think: risk, profile, factors, what-if](#8-think-risk-profile-factors-what-if)
9. [Plan: searching for the easiest changes](#9-plan-searching-for-the-easiest-changes)
10. [Act: the report](#10-act-the-report)
11. [Evaluation on test patients](#11-evaluation-on-test-patients)
12. [What to say in the presentation and paper](#12-what-to-say-in-the-presentation-and-paper)
13. [Glossary](#13-glossary)
14. [Check your understanding](#14-check-your-understanding)
15. [Try it yourself](#15-try-it-yourself)

---

## 1. In simple words (read this first)

Parts 1–3 built separate tools: a model that predicts risk, a GA that found the best 3 features, and four patient groups.
**Part 4 joins them into one assistant** that a nurse or patient could use. You give it the checkup values, and it answers:

```
 YOU GIVE:   age 50, man, 175 cm, 90 kg, BP 135/85, cholesterol level 2, smoker, not active

 AGENT SAYS: Risk 88% (HIGH).
             You are in the "smokers & drinkers" group.
             Main factors: blood pressure, cholesterol, weight.
             Easiest plan: lower BP to 125/80 + cholesterol to normal + become active
                           → risk drops to about 36%.
             Also: stop smoking (medical advice).
             This is not a diagnosis. Talk to a doctor.
```

**It works in three steps, like every agent from Lecture 11:**

```
  1. PERCEIVE   →   2. THINK                          →   3. ACT
  check the         predict risk · find the group          write a report
  input values      search for the easiest plan            a person understands
```

**The surprise we found:** when we asked the model *"what happens if a patient quits smoking?"*, it said the risk
**goes up**. That's wrong medically. It happens because the model learned patterns from this dataset (where the smokers are
mostly younger men), not real causes. An assistant that blindly trusted the model would give **dangerous advice**.

**Our fix:** we added a small **rule book** (a knowledge base) of medical facts. The rules decide which changes are healthy,
and the model only estimates *how much* they help. Mixing learned models with rules is called **neuro-symbolic AI**,
which the project guidelines specifically encourage.

**Part 4 in three sentences:**
1. CardioAgent combines the models from Parts 1–3 into an assistant that perceives, thinks and acts.
2. It **searches** for the easiest set of lifestyle changes that brings a patient's risk below the follow-up level.
3. A rule book stops it from giving advice the data got wrong, like "keep smoking".

---

## 2. Files

```
 results/ga_best_design.json ──┐
 (Part 2: 3-feature model)     │
                               ├──▶ cardio_agent.py ──▶ 04_agent.ipynb ──▶ results/agent_evaluation.json
 results/patient_profiles.json ┤    (the agent)         (explains and       (test results)
 (Part 3: 4 groups)            │                         tests it)
 cardio_utils.py ──────────────┘
```

| File | Role |
|---|---|
| [`cardio_agent.py`](../cardio_agent.py) | The agent itself: the `CardioAgent` class, the knowledge base, the search. Reusable (the Part 5 demo app will import it) |
| [`04_agent.ipynb`](../04_agent.ipynb) | Explains every part step by step and evaluates the agent |
| `results/agent_evaluation.json` | Saved evaluation numbers for the paper |

**Using the agent takes two lines:**

```python
from cardio_agent import CardioAgent
agent = CardioAgent()                 # trains the two models (~2 s)
print(agent.report({"age_years": 50, "is_male": 1, "height_cm": 175, "weight_kg": 90,
                    "ap_hi": 135, "ap_lo": 85, "cholesterol": 2, "gluc": 1,
                    "smoke": 1, "alco": 0, "active": 0}))
```

---

## 3. What is an agent? PEAS

**Lecture 11:** an agent **perceives** its environment through **sensors** and **acts** through **actuators**.
A **rational** agent picks the action expected to do best on its **performance measure**.

```
        ┌──────────── environment ────────────┐
        │  patient / nurse with checkup values │
        └─────┬───────────────────────▲────────┘
     sensors  │ input                 │ report   actuators
              ▼                       │
        ┌──────────────────────────────────────┐
        │  AGENT:  percepts → program → action │
        └──────────────────────────────────────┘
```

### PEAS

| PEAS | CardioAgent |
|---|---|
| **P**erformance | Catch ≥ 80% of sick patients; advice must be **safe** (never unhealthy) and **easy** (lowest effort); understandable |
| **E**nvironment | Routine checkup records: age, sex, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, activity |
| **A**ctuators | A written report: risk level, profile, factors, what-if table, plan, guideline advice, disclaimer |
| **S**ensors | The input record: 11 values, or just 3 for a quick screening |

### What type of agent?

| Agent type (Lecture 11) | CardioAgent |
|---|---|
| Simple reflex (if–then rules) | Partly: input checks and guideline advice |
| Model-based | ✔ the trained MLPs are its model of the world |
| Goal-based | ✔ goal: risk below the follow-up threshold |
| **Utility-based** | ✔ among plans reaching the goal, picks the **lowest effort** |
| Learning | Learning (training, GA) happened **offline** in Parts 1–2 |

**→ CardioAgent is a utility-based agent built on learned models.**

**Environment:** fully observable, episodic (each patient is independent), static, single-agent.

---

## 4. How the agent is built from Parts 1–3

| Component | From | Used for |
|---|---|---|
| **Full model**: MLP (16, 8), 10 features | Part 1 | Full assessment, what-if, planning |
| **Quick model**: MLP (32, 16), 3 features | Part 2 (GA) | Quick screening with only age, BP, cholesterol |
| **4 patient profiles** | Part 3 (k-means) | "You belong to group X" |
| **Thresholds**: full 0.41, quick 0.36 | chosen on validation, recall ≥ 80% | LOW vs. ELEVATED / HIGH |
| **Knowledge base** | new in Part 4 | Input limits, healthy changes, safety rules |
| **Uniform-cost search** | new in Part 4 (Lecture 12) | The easiest plan |

Every earlier part now has a job inside the agent.

---

## 5. The safety check: a surprising finding

Before trusting the model's "what if" answers, we **tested** them. For every validation patient who could make a change,
we applied a **healthy** change and checked whether the model's risk went down (sensible) or up (wrong).

![Safety check](figures/part4_safety.png)

| Healthy change | Patients | Average risk change | Model says risk goes **UP** for |
|---|---|---|---|
| Lower BP by 10/5 | 4,217 | −12.7 points | 1% ✔ |
| Cholesterol down 1 level | 2,568 | −7.7 points | 11% ✔ |
| Become active | 2,010 | −4.8 points | 11% ✔ |
| BMI down 2 | 6,269 | −1.0 point | 6% ✔ |
| **Quit smoking** | 894 | **+1.2 points** | **55%** ✘ |
| **Quit alcohol** | 555 | **+4.1 points** | **71%** ✘ |
| **Glucose down 1 level** | 1,576 | **+2.1 points** | **69%** ✘ |

**Why would a model think quitting smoking is bad?** It learns **associations**, not **causes**:
- Smoking and alcohol are **self-reported**, and the smokers in this dataset are mostly **younger men**
  (the "smokers & drinkers" group from Part 3, with only 46% disease).
- Glucose rises together with cholesterol, so it adds confusing information.

> **Key lesson:** an ML model can be accurate at *predicting* and still be wrong about *what would happen if you changed something*.
> Prediction and advice are different problems.

---

## 6. The knowledge base

```
        LEARNED (neural)                        SYMBOLIC (rules)
 ┌─────────────────────────┐        ┌─────────────────────────────────────┐
 │ MLP: how checkup values │        │ • valid input ranges                │
 │ relate to heart disease │   +    │ • which changes are HEALTHY         │
 │                         │        │ • which model answers to IGNORE     │
 └───────────┬─────────────┘        │ • never show a healthy change as    │
             │                      │   harmful                           │
             │                      └──────────────────┬──────────────────┘
             └──────────────┬──────────────────────────┘
                            ▼
                 safe, explainable advice
```

**The rules:**

| Rule | Detail |
|---|---|
| **Changes the planner may use** (healthy, and the model handles them sensibly) | Lower BP by 10/5 mmHg toward 120/80 (effort 2) · lower cholesterol one level (2) · BMI −2 toward 25 (2) · become active (1) |
| **Guideline-only changes** (recommended, but the model's number is ignored) | Stop smoking · reduce alcohol · bring glucose to normal |
| **Never show a healthy change as harmful** | If the model predicts higher risk after a healthy change, the report says "no measurable benefit in the model" |
| **Input limits** | Same ranges used to clean the data; age outside 29–65 triggers a warning |

The **effort costs** are our own judgement (becoming active is easier than losing weight). They live in the knowledge base and can be changed.

---

## 7. Perceive: checking the input

```
 all 11 values given?  ──yes──▶  FULL assessment
        │ no
        ▼
 age + BP + cholesterol? ──yes──▶  QUICK screening (GA model)
        │ no
        ▼
 refuse: "missing required values"

 any value impossible?    ──▶  refuse, and name the wrong value
 age outside 29–65?       ──▶  continue with a warning
```

| Input | Agent's response |
|---|---|
| Complete record | Full assessment |
| Only age, BP, cholesterol | Quick screening + "add the other values for advice" |
| Cholesterol = 4, BP 80/120 | ✘ "cholesterol must be 1–3" · ✘ "systolic must be higher than diastolic" |
| Age 22 | ⚠ "outside the training ages, less reliable" |
| No blood pressure | ✘ "missing required values: ap_hi" |

**Choosing the model based on the available information** is real agent behaviour. It's also where the GA from Part 2
earns its place: **3 values are enough for a screening.**

---

## 8. Think: risk, profile, factors, what-if

| Output | How it's computed |
|---|---|
| **Risk** | Full MLP → probability. Below the threshold = **LOW**; threshold–70% = **ELEVATED**; ≥ 70% = **HIGH** |
| **Profile** | Nearest k-means centre (Part 3) |
| **Main factors** | Rules: BP ≥ 130, cholesterol above normal, BMI ≥ 25, age ≥ 55 |
| **What-if** | Make each allowed change **in full** (e.g. BP all the way to 120/80), re-run the model |

Example: a patient in the "smokers & drinkers" group (46% disease rate) can still have a personal risk of 88% because of
their blood pressure and cholesterol. **The profile gives context; the MLP gives the personal number.**

---

## 9. Plan: searching for the easiest changes

The what-if table tries one change at a time. The real question is:

> *"What is the **easiest combination** of changes that brings my risk below the follow-up threshold?"*

That's a **search problem** (Lectures 11–12):

| Search element | CardioAgent |
|---|---|
| **State** | how many steps of each change are applied: `(BP, cholesterol, weight, active)` |
| **Initial state** | `(0, 0, 0, 0)`, no changes |
| **Actions** | one more step: BP −10/5 · cholesterol −1 level · BMI −2 · become active |
| **Path cost** | total effort (BP 2, cholesterol 2, weight 2, active 1) |
| **Goal test** | risk < threshold |

```
                         (0,0,0,0)  risk 88%
               ┌──────────────┼──────────────┬───────────────┐
           BP +1 (cost 2)  chol +1 (2)   weight +1 (2)   active (1)
               │              │              │               │
          (1,0,0,0) 61%   (0,1,0,0) 73%  (0,0,1,0) 87%   (0,0,0,1) 82%
               │
              ...  always expand the CHEAPEST plan first ...
               │
          (1,1,0,1)  cost 5  risk 36%  ✔ GOAL
```

### Uniform-cost search (UCS)

- Keeps a **priority queue** of plans, ordered by total cost so far.
- Always expands the **cheapest** plan next.
- Like breadth-first search, but ordered by **cost** instead of number of steps.
- It is **A\* with heuristic h = 0** (Lecture 12).
- With positive costs, **the first plan that reaches the goal is the cheapest**, so UCS is optimal.

**Verified:** for the example patient, brute force checked all 48 combinations (12 reach the goal). The cheapest costs 5:
**exactly the plan UCS found**, after exploring only 14 states.

If no plan reaches the goal (often for older patients), the agent returns the plan with the **largest reduction** and says so.

**Why combinations matter:** for one older patient, each single change barely moves the risk (85% → 84%), but all of them
**together** bring it to 44%. The MLP is non-linear: changes interact.

---

## 10. Act: the report

```
==================================================================
CARDIOAGENT REPORT
==================================================================
Mode: FULL   |   Estimated risk: 88%   |   Level: HIGH  (follow-up threshold 41%)

Patient profile: 'Smokers & drinkers' — in this group 46% of patients have heart disease.
  Your personal estimate is higher than your group's average because of your individual values below.

Main factors:
  • blood pressure 135/85 (low-risk group average 117/77)
  • cholesterol level 2 (normal = 1)
  • BMI 29.4 (overweight range 25–30)

What-if (one change at a time):
  Lower blood pressure from 135/85 to 120/80           88% → 61%
  Lower cholesterol from level 2 to 1 (normal)         88% → 73%
  Become physically active                             88% → 82%
  Lose about 13 kg (BMI 29.4 → 25.0)                   88% → 86%

Suggested plan (easiest set of changes that brings risk below the threshold):
  ✔ Lower blood pressure from 135/85 to 125/80
  ✔ Lower cholesterol from level 2 to 1 (normal)
  ✔ Become physically active
  → estimated risk 88% → 36%   (effort 5, 14 plans searched)

Also recommended by medical guidelines (effect not measurable in this dataset):
  ✔ Stop smoking

This is a decision-support estimate from a model trained on 70,000 checkups.
It is not a diagnosis. Discuss any changes with a doctor.
```

The notebook also shows an older patient (no plan reaches the goal, reported honestly), a healthy patient (LOW risk),
and a quick screening with 3 values.

---

## 11. Evaluation on test patients

All numbers come from the **test set**, which was never used for any decision.

### Does it catch sick patients?

| Mode | Threshold | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Full assessment (10 features) | 0.41 | 0.716 | 0.685 | **0.789** | 0.794 |
| Quick screening (3 features) | 0.36 | 0.710 | 0.677 | **0.790** | 0.789 |

Both catch about **79%** of sick patients, close to the 80% target. The quick mode needs only 3 values.

### How useful are the plans? (1,000 flagged test patients)

| Measure | Result |
|---|---|
| Flagged test patients (risk ≥ threshold) | 5,859 (sample of 1,000 planned) |
| **Plan reaches the goal** | **66%** |
| Average estimated risk | **71% → 39%** |
| Median effort of goal-reaching plans | 4 (e.g. two BP steps, or BP + cholesterol) |
| States explored per patient | 13 on average (milliseconds) |

![Plan analysis](figures/part4_plans.png)

- **Left:** blood pressure is in **72%** of plans, the strongest lever, as every earlier part found.
- **Right:** a sharp age effect. For flagged patients **under 55**, a plan reaches the goal every time; at **55–59**
  about 75%; at **60+** only about 1 in 10. Age can't be changed, and the model weighs it heavily.

### Safety

| Check | Result |
|---|---|
| Plans recommending an unhealthy change (checked on before/after values) | **0 of 1,000** |
| What-if answers the knowledge base had to correct | **112 of 619 (18%)** |

About 1 in 5 what-if answers would have shown a healthy change as *raising* risk. That's more than in the safety check (section 5)
because here each change is made **in full** (e.g. losing 20 kg), creating value combinations that are rare in the data,
where the MLP is less reliable.

> **This is the strongest argument for the neuro-symbolic design: without the rules, the agent would give wrong advice
> for a large share of patients.**

---

## 12. What to say in the presentation and paper

1. **Agent:** CardioAgent is a **utility-based agent** (PEAS above). It perceives a record, predicts risk, explains it with a
   patient profile, and **searches** for the lowest-effort plan below the follow-up threshold.
2. **Safety finding:** the trained models predict that quitting smoking or alcohol *raises* risk, an artefact of associations in the data.
3. **Neuro-symbolic fix:** a knowledge base decides which changes are allowed; the model only estimates their size.
   Zero unhealthy plans; 18% of what-if answers corrected.
4. **Search:** uniform-cost search (A\* with h = 0) finds the provably cheapest plan, verified against brute force.
5. **Uses all parts:** the GA model enables quick 3-value screening (Part 2); the profiles explain each patient (Part 3).
6. **Results:** recall ≈ 79% in both modes; plans reach the goal for 66% of flagged patients (100% under 55, ~11% at 60+);
   average estimated risk falls from 71% to 39%.

**Honest limitations:**
- What-if numbers are **model associations, not proven causal effects**. Every report says so.
- Effort costs are our judgement, not clinical evidence.
- Only 11 checkup values; many real risk factors are missing (family history, ECG, medication).
- Not evaluated by doctors or patients.
- The full and quick modes use different models, so the same patient can get slightly different risk numbers in each mode.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Agent** | Something that perceives its environment and acts on it |
| **PEAS** | Performance, Environment, Actuators, Sensors: the standard way to describe an agent |
| **Utility-based agent** | Chooses the *best* way to reach its goal (here: lowest effort) |
| **Knowledge base** | Hand-written rules the agent uses alongside the learned model |
| **Neuro-symbolic AI** | Combining learned (neural) models with symbolic rules |
| **Association vs. cause** | Two things appearing together in data (association) does not mean one causes the other |
| **What-if analysis** | Changing an input and asking the model how the output changes |
| **State / action / path cost / goal test** | The four parts of a search problem (Lecture 11) |
| **Uniform-cost search** | Always expand the cheapest path so far; optimal with positive costs; A\* with h = 0 |
| **Brute force** | Trying every possibility, used here to verify UCS |
| **Quick screening** | Risk from only 3 values, using the GA model |
| **Follow-up threshold** | Risk level above which a patient is flagged (chosen so ≥ 80% of sick patients are caught) |

---

## 14. Check your understanding

<details>
<summary>1. What are the three steps every agent performs, and what does CardioAgent do in each?</summary>

Perceive (check the input, choose quick or full mode) → think (predict risk, find the profile, search for a plan) →
act (write the report).
</details>

<details>
<summary>2. Why does the model think quitting smoking raises risk?</summary>

It learned associations from this dataset, where smokers are mostly younger men with lower disease rates and smoking is
self-reported. Association is not causation.
</details>

<details>
<summary>3. How does the knowledge base keep the advice safe?</summary>

The planner may only use four healthy changes; smoking, alcohol and glucose advice comes from guidelines, not the model;
and a healthy change is never displayed as raising risk.
</details>

<details>
<summary>4. What are the state, actions, cost and goal in the planning search?</summary>

State: steps applied of each change (BP, cholesterol, weight, active). Actions: one more step of a change. Cost: total
effort. Goal: risk below the threshold.
</details>

<details>
<summary>5. Why is uniform-cost search guaranteed to find the cheapest plan?</summary>

It always expands the cheapest plan so far, and costs are positive, so the first goal state it reaches cannot be beaten by
any plan still in the queue.
</details>

<details>
<summary>6. Why do older patients' plans often fail to reach the goal?</summary>

Age raises the predicted risk and can't be changed, so even after all lifestyle changes the risk can stay above the threshold.
</details>

<details>
<summary>7. Where do Parts 2 and 3 appear inside the agent?</summary>

Part 2's GA model powers the quick 3-value screening; Part 3's profiles describe which group the patient belongs to.
</details>

---

## 15. Try it yourself

| Change | Question to answer |
|---|---|
| Your own values in `agent.report({...})` | Does the plan make sense for you? |
| `be_active` cost 1 → 3 in `MEASURABLE_ACTIONS` | Does the agent stop recommending activity first? |
| Remove `"be_active"` from the planner | How much harder is it to reach the goal? |
| Change the `0.70` HIGH level in `risk_level()` | How does that change how patients are labelled? |

**Next:** Part 5 (optional demo app) and Part 6 (paper and slides).
