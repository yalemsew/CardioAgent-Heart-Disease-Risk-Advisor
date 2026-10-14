"""CardioAgent demo app (Part 5).

Run from the cardio_project folder:
    streamlit run app.py

The app is only a user interface: every number comes from CardioAgent (cardio_agent.py),
the same agent built and tested in 04_agent.ipynb.
"""
import altair as alt
import pandas as pd
import streamlit as st

from cardio_agent import CardioAgent, DISCLAIMER

st.set_page_config(page_title="CardioAgent", page_icon="❤️", layout="wide")

# Example patients for the presentation (same ones used in 04_agent.ipynb)
EXAMPLES = {
    "— enter your own values —": None,
    "50-year-old man, smoker, BP 135/85": dict(age_years=50, is_male=1, height_cm=175, weight_kg=90, ap_hi=135,
                                               ap_lo=85, cholesterol=2, gluc=1, smoke=1, alco=0, active=0),
    "60-year-old woman, BP 160/100, high cholesterol": dict(age_years=60, is_male=0, height_cm=160, weight_kg=85,
                                                            ap_hi=160, ap_lo=100, cholesterol=3, gluc=2, smoke=0,
                                                            alco=0, active=0),
    "40-year-old woman, healthy": dict(age_years=40, is_male=0, height_cm=165, weight_kg=60, ap_hi=115, ap_lo=75,
                                       cholesterol=1, gluc=1, smoke=0, alco=0, active=1),
}
LEVEL_COLOURS = {"LOW": "green", "ELEVATED": "orange", "HIGH": "red"}
LEVELS_123 = {1: "1 — normal", 2: "2 — above normal", 3: "3 — well above normal"}


@st.cache_resource(show_spinner="Starting CardioAgent (training the models, ~3 seconds)...")
def load_agent():
    # Created once, then reused for every click (Streamlit re-runs this script on each interaction).
    return CardioAgent()


agent = load_agent()

# ----------------------------------------------------------------------------- sidebar: SENSORS
st.sidebar.title("❤️ CardioAgent")
st.sidebar.caption("Heart disease risk advisor · CS582 project")

# ?example=1 in the URL opens the app with example 1 loaded (handy for presentations)
try:
    start = min(max(int(st.query_params.get("example", 0)), 0), len(EXAMPLES) - 1)
except ValueError:
    start = 0
example_name = st.sidebar.selectbox("Load an example patient", list(EXAMPLES), index=start)
ex = EXAMPLES[example_name] or dict(age_years=50, is_male=1, height_cm=170, weight_kg=75, ap_hi=120, ap_lo=80,
                                    cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
key = example_name          # new widget keys per example, so loading an example resets the inputs

mode = st.sidebar.radio("Assessment", ["Full (11 values, with advice)", "Quick screening (3 values)"], key="mode")
quick = mode.startswith("Quick")

st.sidebar.subheader("Patient values")
patient = {
    "age_years": st.sidebar.number_input("Age (years)", 18, 100, int(ex["age_years"]), key=f"age{key}"),
    "ap_hi": st.sidebar.number_input("Systolic blood pressure (top number)", 60, 260, int(ex["ap_hi"]), step=5,
                                     key=f"hi{key}"),
    "cholesterol": st.sidebar.selectbox("Cholesterol", [1, 2, 3], index=ex["cholesterol"] - 1,
                                        format_func=LEVELS_123.get, key=f"chol{key}"),
}
if not quick:
    patient["ap_lo"] = st.sidebar.number_input("Diastolic blood pressure (bottom number)", 30, 200,
                                               int(ex["ap_lo"]), step=5, key=f"lo{key}")
    patient["is_male"] = 1 if st.sidebar.radio("Sex", ["Woman", "Man"], index=ex["is_male"], horizontal=True,
                                               key=f"sex{key}") == "Man" else 0
    patient["height_cm"] = st.sidebar.number_input("Height (cm)", 120, 220, int(ex["height_cm"]), key=f"h{key}")
    patient["weight_kg"] = st.sidebar.number_input("Weight (kg)", 30, 220, int(ex["weight_kg"]), key=f"w{key}")
    patient["gluc"] = st.sidebar.selectbox("Glucose (blood sugar)", [1, 2, 3], index=ex["gluc"] - 1,
                                           format_func=LEVELS_123.get, key=f"gluc{key}")
    patient["smoke"] = int(st.sidebar.checkbox("Smokes", bool(ex["smoke"]), key=f"smoke{key}"))
    patient["alco"] = int(st.sidebar.checkbox("Drinks alcohol", bool(ex["alco"]), key=f"alco{key}"))
    patient["active"] = int(st.sidebar.checkbox("Physically active", bool(ex["active"]), key=f"act{key}"))

# ----------------------------------------------------------------------------- PERCEIVE + THINK
result = agent.assess(patient)

# ----------------------------------------------------------------------------- ACT: the report
st.title("CardioAgent: heart disease risk advisor")
st.caption("Enter checkup values on the left. The agent estimates the risk, explains it, and searches for the "
           "easiest lifestyle changes that would lower it.")

for w in result["warnings"]:
    st.warning(w)
if result["errors"]:
    st.error("The agent can't assess these values:\n\n" + "\n".join(f"- {e}" for e in result["errors"]))
    st.stop()

level = result["level"]
c1, c2, c3 = st.columns(3)
c1.metric("Estimated risk", f"{result['risk']:.0%}")
c2.markdown(f"**Risk level**\n\n:{LEVEL_COLOURS[level]}[**{level}**]")
c3.metric("Follow-up threshold", f"{result['threshold']:.0%}",
          help="Chosen so the agent catches at least 80% of patients who have heart disease.")
st.progress(min(result["risk"], 1.0))

if quick:
    st.info("Quick screening uses only age, blood pressure and cholesterol (the 3 features chosen by the genetic "
            "algorithm in Part 2). Switch to **Full** for the patient profile and a lifestyle plan.")
    st.caption(DISCLAIMER)
    st.stop()

left, right = st.columns(2)

with left:
    g = result["profile"]
    st.subheader("Patient profile")
    st.markdown(f"**{g['name']}**: in this group, **{g['disease_rate']:.0%}** of patients have heart disease.")
    if abs(result["risk"] - g["disease_rate"]) >= 0.20:
        direction = "higher" if result["risk"] > g["disease_rate"] else "lower"
        st.caption(f"Your personal estimate is {direction} than the group average because of your individual values.")

with right:
    st.subheader("Main factors")
    if result["factors"]:
        for f in result["factors"]:
            st.markdown(f"- {f}")
    else:
        st.markdown("No major risk factors found in these values.")

st.subheader("What-if: one change at a time")
if result["what_if"]:
    wf = pd.DataFrame(
        [{"change": "Current values", "risk": result["risk"], "kind": "current"}]
        + [{"change": text, "risk": risk, "kind": "no benefit in model" if corrected else "change"}
           for text, risk, corrected in result["what_if"]]
    )
    wf["label"] = wf["risk"].map("{:.0%}".format)
    colours = alt.Scale(domain=["current", "change", "no benefit in model"], range=["#d62728", "#1f77b4", "#bbbbbb"])
    base = alt.Chart(wf).encode(
        y=alt.Y("change:N", sort=None, title=None, axis=alt.Axis(labelLimit=400, labelFontSize=13)),
        x=alt.X("risk:Q", title="Estimated risk", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    )
    bars = base.mark_bar(height=22).encode(color=alt.Color("kind:N", scale=colours, legend=None))
    labels = base.mark_text(align="left", dx=4, fontSize=13).encode(text="label:N")
    rule = (alt.Chart(pd.DataFrame({"t": [result["threshold"]]}))
            .mark_rule(strokeDash=[4, 4], color="black").encode(x="t:Q"))
    st.altair_chart((bars + labels + rule).properties(height=38 * len(wf)), width="stretch")
    st.caption("Dashed line = follow-up threshold."
               + (" Grey bars: the model predicted a rise after a healthy change, which the agent's medical rules "
                  "do not allow it to display, so it shows no benefit instead."
                  if (wf["kind"] == "no benefit in model").any() else ""))
else:
    st.markdown("No lifestyle changes apply: blood pressure, cholesterol and weight are already at target, "
                "and the patient is active.")

st.subheader("Suggested plan")
p = result["plan"]
if level == "LOW":
    st.success("Risk is below the follow-up threshold. Keep up the healthy habits.")
elif not p["steps"]:
    st.info("No measurable lifestyle change is available for this patient.")
else:
    if p["reached_goal"]:
        st.markdown("The **easiest** set of changes that brings the estimated risk below the follow-up threshold:")
    else:
        st.warning("No combination of lifestyle changes brings the risk below the threshold (age and other fixed "
                   "factors weigh in). This plan gives the **largest** reduction:")
    for step in p["steps"]:
        st.markdown(f"✔ {step}")
    a, b, c = st.columns(3)
    a.metric("Risk after the plan", f"{p['risk']:.0%}", f"{(p['risk'] - result['risk']) * 100:+.0f} points",
             delta_color="inverse")
    b.metric("Effort score", p["cost"], help="BP step 2 · cholesterol step 2 · weight step 2 · become active 1")
    c.metric("Plans searched", p["states_explored"], help="Number of states explored by the uniform-cost search")

if result["guideline_advice"]:
    st.subheader("Also recommended by medical guidelines")
    for advice in result["guideline_advice"]:
        st.markdown(f"✔ {advice}")
    st.caption("This dataset cannot measure the effect of these changes, so the agent recommends them from its "
               "medical rules rather than from the model.")

st.divider()
st.caption(DISCLAIMER)

with st.expander("How does CardioAgent work?"):
    st.markdown(
        """
| Step | What happens | Project part |
|---|---|---|
| **Perceive** | Check the values are valid; choose full or quick mode | Part 4 |
| **Risk** | Neural network (MLP) trained on 48,000 checkups | Part 1 (full) · Part 2 GA model (quick) |
| **Profile** | Nearest of 4 patient groups found by k-means | Part 3 |
| **Plan** | Uniform-cost search for the lowest-effort changes that bring risk below the threshold | Part 4 |
| **Safety rules** | Only healthy changes allowed; smoking, alcohol and glucose advice comes from guidelines | Part 4 |

On held-out test patients, the agent catches about **79%** of people with heart disease.
"""
    )
