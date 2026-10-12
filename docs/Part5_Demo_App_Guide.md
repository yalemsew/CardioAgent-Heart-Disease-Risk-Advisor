# Part 5: The CardioAgent Demo App

**Project:** CardioAgent, heart disease risk prediction (CS582 Machine Learning)
**App code:** [`app.py`](../app.py) (built with Streamlit) · **Uses:** [`cardio_agent.py`](../cardio_agent.py) from Part 4

---

## 1. In simple words (read this first)

Part 4 built the agent, but you could only use it by writing Python code in a notebook.
**Part 5 gives it a simple web page**: you type a patient's values into a form, and the report appears instantly.
It's the same agent with the same numbers, just easier to use and **much better for the presentation**.

```
  ┌──────────────── web page in your browser ─────────────────┐
  │  FORM (left)          │   REPORT (right)                  │
  │  age, BP, cholesterol │   risk 88% · HIGH                 │
  │  height, weight, ...  │   profile · main factors          │
  │  ☑ smokes             │   what-if chart                   │
  │                       │   suggested plan · guideline advice│
  └────────────────────────────────────────────────────────────┘
         change any value → the report updates immediately
```

**The app doesn't contain any machine learning itself.** It only shows the form and draws the report.
Every number comes from `CardioAgent`, which was built and tested in Part 4.

---

## 2. How to run it

1. Make sure the libraries are installed (once): from inside the `cardio_project` folder run `pip install -r requirements.txt`.
2. Open a **Terminal** (in VS Code: menu **Terminal → New Terminal**).
3. Go to the project folder (wherever you saved it) and start the app:

   ```bash
   cd cardio_project
   streamlit run app.py
   ```

   If the `streamlit` command isn't found, try `python -m streamlit run app.py`.

4. Your browser opens at **http://localhost:8501**. The first load takes about 3 seconds while the models train.
5. To stop the app, click in the Terminal and press **Ctrl + C**.

**Shortcut for presentations:** add `?example=1`, `?example=2` or `?example=3` to the address to open the app with an example
patient already loaded, for example **http://localhost:8501/?example=1**.

---

## 3. What the screen shows

![App with example 1](figures/part5_app_example1.png)

| Area | What it shows | Comes from |
|---|---|---|
| **Sidebar: example patient** | Loads one of three ready-made patients | — |
| **Sidebar: assessment** | Full (11 values, with advice) or Quick (3 values) | Part 4, perceive step |
| **Sidebar: patient values** | The form. Changing any value updates the report | — |
| **Estimated risk / level** | The risk percentage and LOW / ELEVATED / HIGH | Part 1 MLP (full) or Part 2 GA model (quick) |
| **Follow-up threshold** | Risk above which a patient is flagged (catches ≥ 80% of sick patients) | Parts 1–2 |
| **Patient profile** | Which of the 4 groups the patient belongs to | Part 3 k-means |
| **Main factors** | Values that raise the risk | Part 4 rules |
| **What-if chart** | Risk after each change on its own. Dashed line = threshold. Grey bars = the knowledge base corrected the model | Part 4 |
| **Suggested plan** | The easiest set of changes that brings the risk below the threshold, with effort score and number of plans searched | Part 4 uniform-cost search |
| **Medical guidelines** | Advice on smoking, alcohol and glucose, which the dataset can't measure | Part 4 knowledge base |
| **How does it work?** | A short table linking each feature to the project parts | — |

### When no plan reaches the goal

![App with example 2](figures/part5_app_example2.png)

For this 60-year-old patient, even all the changes together only bring the risk to 44%, just above the 41% threshold.
The app says so honestly in a yellow box. The grey bars show two changes where the model's answer was corrected by the
medical rules.

### Invalid input

If you enter impossible values (for example diastolic higher than systolic), the app shows a red box explaining what's wrong
instead of a risk. This is the agent's **perceive** step at work.

---

## 4. How the code is organised

```python
agent = load_agent()            # create CardioAgent once (cached; Streamlit re-runs the script on every click)
patient = {...}                 # read the sidebar form
result = agent.assess(patient)  # perceive → think → plan   (all the ML happens here)
# ... then draw result["risk"], result["profile"], result["what_if"], result["plan"], ...
```

| Part of `app.py` | Job |
|---|---|
| `load_agent()` with `@st.cache_resource` | Trains the models **once**, not on every click |
| Sidebar section | The form (the agent's **sensors**) |
| `agent.assess(patient)` | All the thinking, done by Part 4's code |
| Main section | Draws the report (the agent's **actuators**) |

Streamlit is a Python library that turns a script into a web page: `st.metric`, `st.markdown`, `st.altair_chart` and so on
each add one element to the page.

---

## 5. Testing

The app was tested in two ways:

1. **Automated tests** with Streamlit's testing tool, which simulates clicks: all three examples, quick mode, and invalid
   blood pressure. No errors, and the numbers match the notebook exactly (88%, 85%, 10%; quick mode 9%).
2. **Screenshots** in a real browser (the images above), to check that the layout and chart labels are readable.

---

## 6. Demo script for the presentation (about 3 minutes)

| Step | Do | Say |
|---|---|---|
| 1 | Open `?example=1` | "A 50-year-old male smoker with BP 135/85. The agent estimates 88%, HIGH." |
| 2 | Point at the profile | "Part 3's unsupervised learning places him in the 'smokers & drinkers' group, 46% disease; his personal risk is higher because of his blood pressure and cholesterol." |
| 3 | Point at the what-if chart | "Lowering blood pressure alone would bring him to 61%." |
| 4 | Point at the plan | "Uniform-cost search found the easiest combination: three changes bring him below the threshold, to 36%, after checking only 14 plans." |
| 5 | Point at "Stop smoking" | "Our safety check found the model thinks quitting smoking *raises* risk, an artefact of the data. The knowledge base gives this advice from medical guidelines instead." |
| 6 | Switch to example 2 | "For a 60-year-old, no plan reaches the goal, and the agent says so. Grey bars show where the rules corrected the model." |
| 7 | Switch to **Quick screening** | "With only age, blood pressure and cholesterol, the GA model from Part 2 still screens the patient." |
| 8 | Type an impossible value | "Invalid input is rejected with a clear message." |

**Tip:** start the app **before** the presentation, so the 3-second startup doesn't happen live, and keep the notebook
open as a backup.

---

## 7. Check your understanding

<details>
<summary>1. Does the app contain its own machine-learning model?</summary>

No. It only shows the form and the report. All predictions come from `CardioAgent` in `cardio_agent.py`.
</details>

<details>
<summary>2. Why is <code>load_agent()</code> cached?</summary>

Streamlit re-runs the whole script every time you change a value. Without caching, the models would be retrained on every click.
</details>

<details>
<summary>3. Which project part produces each number on the screen?</summary>

Risk: Part 1 MLP (full) or Part 2 GA model (quick). Profile: Part 3. Plan, what-if, safety rules: Part 4.
</details>
