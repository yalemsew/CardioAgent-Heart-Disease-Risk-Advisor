# CardioAgent: Project Workflow

*How the project goes from the raw dataset to the final demo app: which algorithm is used where, why, which files
are involved, and what results each step produced. All numbers come from the saved notebook outputs and the
`results/*.json` files.*

---

## 1. The project in one paragraph

CardioAgent predicts a person's risk of cardiovascular disease from routine checkup values (age, sex, height, weight,
blood pressure, cholesterol, glucose, smoking, alcohol, physical activity). It uses the Kaggle Cardiovascular Disease
dataset (70,000 patients; 68,559 after cleaning). The predictor is an **MLP neural network**. Around it, an
**intelligent agent** checks the input, explains the result using **patient profiles**, and searches for the **easiest
lifestyle changes** that would lower the risk. A **Streamlit app** gives the agent a user interface.
It is decision support, **not** a diagnosis.

---

## 2. The whole workflow at a glance

```
data/cardio_train.csv
        │
        ▼
 cardio_utils.py ── clean → add features → split 70/15/15 → scale        (shared by everything)
        │
        ├──▶ Part 1  01_cardio_basics.ipynb
        │       Train MLP (16, 8) · compare with SVM, Random Forest, Naive Bayes
        │       Choose the decision threshold (catch ~80% of sick patients)
        │       → design of the FULL model (written directly into cardio_agent.py)
        │
        ├──▶ Part 2  02_genetic_algorithm_design.ipynb
        │       Genetic algorithm searches features + network size
        │       → results/ga_best_design.json   (design of the QUICK model)
        │
        ├──▶ Part 3  03_patient_profiles.ipynb
        │       PCA, LDA, SOM (analysis) · K-means (4 patient profiles)
        │       → results/patient_profiles.json
        │
        ▼
 Part 4  cardio_agent.py  +  04_agent.ipynb
        Trains both MLPs from the CSV · loads the two JSON files
        Perceive → Think (predict, profile, plan with uniform-cost search) → Act (report)
        → results/agent_evaluation.json   (evaluation report only)
        │
        ▼
 Part 5  app.py  (Streamlit user interface)
```

**Notebooks = experiments** (slow; they explore, compare and decide) → they **save their conclusions as JSON** →
**`cardio_agent.py` / `app.py` = the product** (fast, fixed, repeatable).

---

## 3. Step by step

### Step 0: Data preparation ([`cardio_utils.py`](../cardio_utils.py))

Every notebook and the agent prepare the data with the same two functions:

| Function | What it does |
|---|---|
| `load_clean_data()` | Loads the CSV and removes impossible values (e.g. blood pressure of −150 or 16,020), about 2% of rows. Adds `age_years`, `bmi` and `is_male`. Result: 68,559 patients, 10 features + the label `cardio`. |
| `split_and_scale()` | Stratified split: **70% train / 15% validation / 15% test**. A `StandardScaler` is fitted on the training set only, to avoid data leakage. |

- **Validation set:** used to make choices (model settings, threshold).
- **Test set:** used **once** at the end to report honest results.

---

### Step 1: The predictor and the comparison ([`01_cardio_basics.ipynb`](../01_cardio_basics.ipynb))

**Main model:** `MLPClassifier`, a feed-forward neural network trained with backpropagation:
- two hidden layers (16 and 8 neurons) with ReLU;
- early stopping to prevent overfitting;
- it outputs the probability of heart disease.

**Comparison:** three other algorithms from the course were trained on the same data.

| Model (validation set, threshold 0.5) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **MLP (16, 8)** | 0.735 | 0.741 | **0.713** | **0.727** | 0.803 |
| Random Forest | 0.737 | 0.764 | 0.677 | 0.718 | 0.804 |
| SVM (RBF), trained on a 10,000-row sample | 0.737 | 0.748 | 0.705 | 0.726 | 0.794 |
| Naive Bayes | 0.713 | 0.763 | 0.609 | 0.678 | 0.781 |

**What this showed:**
- MLP, SVM and Random Forest are essentially tied at about 73%. When very different algorithms hit the same ceiling,
  the limit is **in the data**, not the algorithm.
- Naive Bayes is weaker, because its assumption that features are independent is wrong here (the two blood pressure
  values are strongly related).
- Random Forest feature importance ranks blood pressure, age, cholesterol and BMI highest.

**Why the MLP was kept.** It did not clearly win; the gaps are tiny. It was kept because:
1. it has the best recall and F1 of the four, and catching sick patients matters most;
2. the genetic algorithm in Part 2 designs MLPs, so the rest of the project builds on it;
3. the SVM is too slow on the full data;
4. an MLP gives smooth probabilities, which suits the what-if planner (a Random Forest's output changes in jumps).
   *This last point is our own reasoning; the notebook doesn't state it.*

**Threshold choice.** Missing a sick patient is worse than a false alarm. So the decision threshold was lowered from
0.5 to the highest value that still catches at least 80% of sick patients on the validation set.

| MLP on the TEST set | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Threshold 0.5 | 0.733 | 0.741 | 0.708 | 0.724 | 0.794 |
| **Threshold 0.4 (chosen)** | 0.712 | 0.677 | **0.797** | 0.732 | 0.794 |

> SVM, Random Forest and Naive Bayes are **for comparison only**. They are not used anywhere after Part 1.

---

### Step 2: Genetic algorithm designs the quick model ([`02_genetic_algorithm_design.ipynb`](../02_genetic_algorithm_design.ipynb))

**Question:** which inputs does the model really need, and how big should the network be? There are 8,192 possible
combinations.

**Method:** a genetic algorithm, which copies evolution.
- Each design is a 13-bit chromosome: 10 bits switch the features on or off, and 3 bits choose the network shape.
- Fitness = validation ROC-AUC − 0.002 × (number of features). The small penalty means that when two designs score
  the same, the one with fewer features wins. Recall is not used as fitness, because a model that says "disease" for
  everyone has 100% recall; recall is tuned later with the threshold.
- Settings: population 20, 15 generations, elitism, crossover and mutation.

| Result | Value |
|---|---|
| Designs trained | 191 of 8,192 (about 136 s) |
| **Best design** | **3 features (systolic BP, cholesterol, age) · network (32, 16)** |
| Best fitness | 0.7938 (validation AUC ≈ 0.80 minus the 3-feature penalty) |
| Random search, same budget | 0.7903 (the GA did only slightly better; many designs are nearly as good) |
| AUC over 5 seeds | GA design 0.7995 ± 0.0002 vs Part 1 design 0.8009 ± 0.0015 |
| Test set (threshold 0.36) | Accuracy 0.710 · Precision 0.677 · Recall 0.790 · AUC 0.789 |

**Conclusion:** practically the same performance with **70% fewer inputs**.

**Saved to:** [`results/ga_best_design.json`](../results/ga_best_design.json). This is the *recipe* for the quick
model (which columns and what network size), **not** trained weights and **not** training data.

---

### Step 3: Patient profiles with unsupervised learning ([`03_patient_profiles.ipynb`](../03_patient_profiles.ipynb))

| Algorithm | Purpose | Result |
|---|---|---|
| **PCA** | A 2-D picture of the 10-D data | 2 components keep 38% of the variation. Sick and healthy patients overlap heavily, which explains the ~73% ceiling. |
| **LDA** | The best single direction separating sick from healthy | Built mostly from systolic BP, age and cholesterol, the **same 3 features the GA chose**. AUC 0.795. |
| **K-means (k = 4)** | Group similar patients **without** using the diagnosis | 4 named profiles (below). Silhouette is about 0.2, so the groups are soft regions, not sharp clusters; k = 4 was chosen for interpretability. |
| **SOM (10×10)** | Clustering with a self-organising neural network | Found the same high- and low-risk regions as k-means, which confirms the profiles are real. |

| K-means profile | Patients (train) | Heart disease rate |
|---|---|---|
| High blood pressure & overweight | 13,158 | **79%** |
| High cholesterol & sugar | 4,592 | **62%** |
| Smokers & drinkers | 4,494 | **46%** |
| Lower-risk baseline | 25,747 | **33%** |

**Do the profiles predict well on their own?** Using only a patient's group gives AUC 0.71, against about 0.80 for
the MLP. So the profiles are used as an **explanation layer**, not as a predictor.

**Saved to:** [`results/patient_profiles.json`](../results/patient_profiles.json): the group centres, names, disease
rates and average values.

> PCA, LDA and SOM are **analysis for the report only**. K-means is used by the final agent.

---

### Step 4: The intelligent agent ([`cardio_agent.py`](../cardio_agent.py), tested in [`04_agent.ipynb`](../04_agent.ipynb))

The `CardioAgent` class follows a **perceive → think → act** cycle.

**At startup (`__init__`):**
1. Loads and splits the CSV with `cardio_utils.py`.
2. Trains the **full MLP**: 10 features, (16, 8). Its design is written directly in the code.
3. Trains the **quick MLP** with the features and shape read from `ga_best_design.json`.
4. Picks a threshold for each model on the validation set, so it catches about 80% of sick patients
   (full model 0.41, quick model 0.36).
5. Loads the k-means centres and group information from `patient_profiles.json`.

Both models are **retrained from the CSV every time** (about 3 seconds). No trained model file is saved.

**For each patient:**

| Stage | Method | What happens |
|---|---|---|
| **Perceive** | `perceive()` | Checks values against valid ranges. Chooses **full** mode (all values given) or **quick** mode (only age, BP, cholesterol). Warns if the age is outside the training range. |
| **Think: predict** | Full or quick MLP | Gives the risk %, and a level: LOW / ELEVATED / HIGH. |
| **Think: explain** | `profile()`, nearest k-means centre | "You are in group X, where Y% have heart disease." Lists risk factors compared with the lower-risk group. |
| **Think: what-if** | Full MLP | The new risk after each single lifestyle change. |
| **Think: plan** | `plan()`, **uniform-cost search** | Finds the cheapest combination of changes (lower BP, lower cholesterol, lose weight, become active) that brings the risk below the threshold. |
| **Act** | `assess()` / `report()` | Returns the result with a disclaimer. |

**Knowledge base (medical rules).** A safety check showed that this dataset makes the model predict a *higher* risk
after some clearly healthy changes:

| Healthy change | % of patients where the model says risk goes UP |
|---|---|
| Quit smoking | 55% |
| Lower glucose | 69% |
| Quit alcohol | 71% |

So the agent recommends those three changes **from medical guidelines**, never from the model's number. It also never
displays a healthy change as harmful.

**Planner evaluation** (1,000 at-risk test patients):

| Result | Value |
|---|---|
| Plans that bring the risk below the threshold | 66.4% |
| Average risk before → after | 70.5% → 38.9% |
| Median plan cost | 4 effort units |
| Most-used action | Lower blood pressure (72% of plans) |
| Plans that suggest an unhealthy change | 0 |

**Saved to:** [`results/agent_evaluation.json`](../results/agent_evaluation.json). This is an evaluation report; the
app doesn't use it.

---

### Step 5: The demo app ([`app.py`](../app.py))

A Streamlit user interface. **It contains no machine learning itself.** It collects the patient's values in the
sidebar, calls `agent.assess(patient)`, and shows the risk, the profile, the what-if chart and the plan.

| App mode | Model used | Inputs |
|---|---|---|
| **Full** (11 values, with advice) | Full MLP (16, 8) | 10 features (height and weight become BMI) |
| **Quick screening** (3 values) | Quick MLP (32, 16), designed by the GA | Systolic BP, cholesterol, age |

**Files the app depends on:**

```
app.py                                  ← user interface
 └─ cardio_agent.py                     ← the agent: trains the MLPs, assesses, plans
     ├─ cardio_utils.py                 ← data cleaning, split, scaling
     │   └─ data/cardio_train.csv       ← training data for both MLPs
     ├─ results/ga_best_design.json     ← quick model's features + network size
     └─ results/patient_profiles.json   ← k-means profiles (explanation only)
```

Not used by the app: the notebooks, `results/agent_evaluation.json` and `docs/`.

---

## 4. Role of every algorithm

| Algorithm | Where | Role in the final system |
|---|---|---|
| **MLP neural network** | Part 1, `cardio_agent.py` | **The predictor**: gives the risk number (full and quick models) |
| **Genetic algorithm** | Part 2 | **Designed the quick MLP** (inputs and network size) |
| **K-means** | Part 3, `cardio_agent.py` | **Explanation**: tells the user which patient group they belong to |
| **Uniform-cost search** (classic AI, not ML) | `cardio_agent.py` | **Planning**: the cheapest lifestyle plan, using the MLP's predictions |
| SVM, Random Forest, Naive Bayes | Part 1 | Comparison only, to justify the MLP |
| PCA, LDA, SOM | Part 3 | Analysis only, to understand the data and confirm the GA and k-means results |

**None of the other algorithms feed inputs into the MLP.** The MLP only sees the patient's own (scaled) checkup
values:

- `ga_best_design.json` affects the quick MLP only as **settings** (which columns, what size).
- `patient_profiles.json` is used only **after** the prediction, to explain it.

---

## 5. Why the JSON files instead of rerunning the steps

| File | Why it's saved rather than recomputed in the app |
|---|---|
| `ga_best_design.json` | The GA takes about 136 s, so the app would take over 2 minutes to start. A rerun might also pick a slightly different design, and the app would no longer match the report. |
| `patient_profiles.json` | The group names were assigned after inspecting the groups. Cluster numbers can change between runs, so the names and disease rates could get mixed up. |

---

## 6. How to run

```powershell
pip install -r requirements.txt     # once
streamlit run app.py                # opens http://localhost:8501  (add ?example=1 for an example patient)
```

Optionally, rerun the notebooks **in order** (01 → 02 → 03 → 04). Part 2 and Part 3 regenerate the JSON files the
agent needs. The JSON files are already in the repository, so the app runs without the notebooks.

To use the agent without the app:

```python
from cardio_agent import CardioAgent
agent = CardioAgent()
print(agent.report(dict(age_years=50, is_male=1, height_cm=175, weight_kg=90, ap_hi=135,
                        ap_lo=85, cholesterol=2, gluc=1, smoke=1, alco=0, active=0)))
```

---

*Disclaimer: CardioAgent is a decision-support estimate from a model trained on 70,000 checkups. It is not a
diagnosis.*
