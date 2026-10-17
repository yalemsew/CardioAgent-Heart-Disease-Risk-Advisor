# CardioAgent: Team Plan (5 members)

How we split the project so each person **masters one section**, presents it, and writes it up for the paper,
while everyone still understands the whole project.

> **Everyone first:** read [`Project_Overview.md`](Project_Overview.md) (about 15 minutes). It explains the whole project in plain language.
> Then fill in your name below and start on your section.

---

## 1. The split at a glance

| Role | Owner | Section | Course topics | Workload |
|---|---|---|---|---|
| **Person 1: Problem & Data** | *(name)* | Problem, dataset, cleaning, exploration, data split · **final results & conclusion** | L6 | ●●●○ |
| **Person 2: Prediction Models** | *(name)* | Part 1 models: MLP, SVM, Random Forest, Naive Bayes, metrics, threshold | L4–6, L8, L15 | ●●●○ |
| **Person 3: Genetic Algorithm** | *(name)* | Part 2: GA designs the model | L11–13 | ●●●○ |
| **Person 4: Patient Profiles** | *(name)* | Part 3: PCA, LDA, k-means, SOM | L7, L9–10 | ●●●○ |
| **Person 5: Agent & Demo** | *(name)* | Part 4: agent, safety rules, search · Part 5: live demo | L11–12 | ●●●● |

Person 5's section is the largest, so Person 1 also takes the **final results, limitations and conclusion** slides
to balance it.

```
 Person 1          Person 2           Person 3          Person 4           Person 5
 ─────────         ─────────          ─────────         ─────────          ─────────
 problem &   ──▶   models learn  ──▶  GA picks the ──▶  find 4 patient ──▶ agent puts it all
 data              to predict         3 key features    groups             together + live demo
                                                                                 │
 Person 1 closes: final results, limitations, conclusion  ◀──────────────────────┘
```

---

## 2. Each role in detail

### Person 1: Problem & Data

**You own:** the problem, the dataset and how it was prepared. You **open and close** the presentation.

**Study:**
- [`01_cardio_basics.ipynb`](../01_cardio_basics.ipynb), Steps 1–7 (load, inspect, clean, feature engineering, charts, split, scaling)
- [`cardio_utils.py`](../cardio_utils.py) (short: the same cleaning, split and scaling, used by every other part)
- [`Project_Overview.md`](Project_Overview.md), chapters 1–2 and 8–9

**Concepts to master:**
- The problem and why early risk detection matters; decision support vs. diagnosis
- The 11 checkup features and the target (`cardio`)
- Data cleaning: impossible values (blood pressure −150 or 16,020) and the rules used to remove them
- Feature engineering: age in years, BMI, `is_male`
- Exploratory charts: which features relate to disease (blood pressure, age, cholesterol, BMI)
- Train / validation / test split, stratification, and why the test set is used only once
- Scaling, and why it's fitted on training data only (data leakage)

**Key numbers:** 70,000 → 68,559 patients (2.1% removed) · 50/50 sick vs. healthy · split 47,991 / 10,284 / 10,284 ·
strongest correlations: `ap_hi` 0.43, `ap_lo` 0.34, age 0.24, cholesterol 0.22, BMI 0.19

**Slides (about 4):** problem & motivation · dataset · cleaning & feature engineering · *(at the end)* final results,
limitations & conclusion

**Paper sections:** Introduction · Dataset & preprocessing · Conclusion & future work

**Practice questions:**
1. Why did we remove rows instead of fixing them?
2. Why is BMI more useful than height and weight separately?
3. Why do we need three data splits, not two?
4. Why is the scaler fitted only on training data?
5. Why is this a decision-support tool and not a diagnosis tool?

---

### Person 2: Prediction Models

**You own:** how the computer learns to predict, which model was chosen, and why the threshold was lowered.

**Study:**
- [`01_cardio_basics.ipynb`](../01_cardio_basics.ipynb), Steps 8–12 (MLP, evaluation, model comparison, threshold, prediction)
- [`Project_Overview.md`](Project_Overview.md), chapter 3

**Concepts to master:**
- Supervised learning
- How an MLP learns: forward pass, error, backpropagation, epochs, early stopping; ReLU and Adam vs. the lecture versions
- The 3 comparison models: SVM (margin, RBF kernel), Random Forest (trees + voting, feature importance), Naive Bayes (independence assumption)
- Confusion matrix, accuracy, precision, recall, F1, ROC-AUC
- Why all strong models reach ~73% (the limit is in the data)
- The threshold trade-off: 50% → 41% raises recall from 71% to 79% at the cost of more false alarms

**Key numbers:** accuracy: RF 0.737 · MLP 0.735 · SVM 0.737 · NB 0.713 · MLP ROC-AUC 0.80 ·
test recall 71% (threshold 0.50) → 79% (threshold 0.41) · ≈ 410 more sick people caught, ≈ 590 more false alarms

**Slides (about 4):** how a neural network learns · four models compared · metrics: why recall matters · the threshold trade-off

**Paper sections:** Prediction models · Evaluation metrics · Model comparison · Threshold selection

**Practice questions:**
1. What does "backpropagation" do, in plain words?
2. Why did all models get stuck near 73%?
3. Why is accuracy alone not enough in medicine?
4. Why did accuracy *drop* when we lowered the threshold, and why is that acceptable?
5. Why did we keep the MLP rather than Random Forest, which scored the same?

---

### Person 3: Genetic Algorithm

**You own:** how the model design was found automatically.

**Study:**
- [`02_genetic_algorithm_design.ipynb`](../02_genetic_algorithm_design.ipynb)
- [`Part2_Genetic_Algorithm_Guide.md`](Part2_Genetic_Algorithm_Guide.md) (diagrams, glossary, self-check questions)
- `results/ga_best_design.json`

**Concepts to master:**
- Why choosing features and network size is a **search problem** (8,192 designs; backprop can't do it)
- Chromosome encoding: 13 bits (10 feature bits + 3 shape bits)
- Fitness = validation AUC − 0.002 × number of features, and why not recall
- Roulette selection, one-point crossover, mutation (0.05), elitism
- Exploration vs. exploitation
- The two honesty checks: random search with the same budget, and 5 training seeds
- GA (designs the network) vs. backpropagation (learns the weights)

**Key numbers:** 191 of 8,192 designs tried · ~2 minutes · winner: `ap_hi`, `cholesterol`, `age_years`, shape (32, 16) ·
validation AUC 0.800 (3 features) vs 0.801 (10 features) · GA 0.794 vs random search 0.790 · test recall 79%

**Slides (about 4):** the design problem · chromosome & fitness · GA loop & progress chart · result: 3 features are enough

**Paper sections:** GA method · GA experiments & results · comparison with random search

**Practice questions:**
1. What does the chromosome `[1,1,0,0,0,0,0,1,1,0, 1,0,1]` mean?
2. Why is AUC the fitness, not recall?
3. Why do weaker designs still get selected sometimes?
4. Is the GA really better than random guessing? How do you know?
5. If smoking was dropped, does that mean smoking is harmless?

---

### Person 4: Patient Profiles

**You own:** the unsupervised analysis that finds patient types and explains the accuracy ceiling.

**Study:**
- [`03_patient_profiles.ipynb`](../03_patient_profiles.ipynb)
- [`Part3_Patient_Profiles_Guide.md`](Part3_Patient_Profiles_Guide.md)
- `results/patient_profiles.json`

**Concepts to master:**
- Supervised vs. unsupervised learning
- PCA: 10 features → 2-D picture; what PC1 and PC2 mean; only 38% of information kept
- LDA: best separating direction; same top features as the GA; AUC 0.795 ≈ MLP → patterns are mostly linear
- K-means: the algorithm, choosing k (elbow, silhouette), why k = 4
- SOM: competitive learning, winner neuron, neighbourhood function; why it agrees with k-means
- Why the profiles are an **explanation layer**, not a predictor (AUC 0.71)

**Key numbers:** 4 groups: high BP & overweight **79%** · high cholesterol & sugar **62%** · smokers & drinkers **46%** ·
lower-risk **33%** · silhouette ≈ 0.2 · profile-only AUC 0.709 · LDA AUC 0.795

**Slides (about 4):** why look for groups · PCA picture (explains the 73% ceiling) · the 4 profiles · SOM confirms k-means

**Paper sections:** Unsupervised analysis · patient profiles · discussion of the accuracy ceiling

**Practice questions:**
1. K-means never saw who was sick. How can its groups have different disease rates?
2. Why must the data be scaled before k-means?
3. Why k = 4 if the silhouette scores were all similar?
4. What's the difference between PCA and LDA?
5. What does the SOM's neighbourhood function do?

---

### Person 5: Agent & Demo

**You own:** the intelligent agent (the project's main novelty) and the live demo.

**Study:**
- [`04_agent.ipynb`](../04_agent.ipynb)
- [`cardio_agent.py`](../cardio_agent.py) (the agent code)
- [`app.py`](../app.py) and [`Part5_Demo_App_Guide.md`](Part5_Demo_App_Guide.md) (includes an 8-step demo script)
- [`Part4_Agent_Guide.md`](Part4_Agent_Guide.md)

**Concepts to master:**
- Agent, PEAS, agent types; why CardioAgent is utility-based
- Perceive → think → act; quick vs. full mode
- **The safety finding:** the model predicts quitting smoking *raises* risk (association, not cause)
- The knowledge base and **neuro-symbolic AI**
- The planning search: state, actions, cost, goal; **uniform-cost search** (A\* with h = 0); why it's optimal
- The evaluation: recall, goal-reached rate by age, 0 unhealthy plans, 18% what-if corrections

**Key numbers:** thresholds 0.41 (full) / 0.36 (quick) · plans reach the goal for **66%** (100% under 55, ~11% at 60+) ·
risk 71% → 39% · **0** unhealthy plans · **18%** what-if answers corrected · UCS = brute force (cost 5, 14 of 48 states)

**Slides (about 5):** what is an agent + PEAS · the safety surprise · knowledge base · the planning search · evaluation
· **then the live demo (about 3 minutes)**

**Paper sections:** Agent design · knowledge base · planning search · agent evaluation · demo application

**Practice questions:**
1. Why does the model think quitting smoking raises risk?
2. How does the knowledge base keep the advice safe?
3. What are the state, actions, cost and goal in the search?
4. Why is uniform-cost search guaranteed to find the cheapest plan?
5. Why do older patients' plans often fail to reach the goal?

---

## 3. Presentation outline (about 25 minutes + questions)

Adjust the times if the instructor gives a different length.

| # | Section | Presenter | Time |
|---|---|---|---|
| 1 | Problem, dataset, cleaning | Person 1 | 3 min |
| 2 | Prediction models & threshold | Person 2 | 4 min |
| 3 | Genetic algorithm designs the model | Person 3 | 4 min |
| 4 | Patient profiles (unsupervised) | Person 4 | 4 min |
| 5 | The agent: PEAS, safety, search | Person 5 | 5 min |
| 6 | **Live demo** | Person 5 | 3 min |
| 7 | Results, limitations, conclusion | Person 1 | 2 min |
| — | Questions | Everyone (owner answers first) | — |

**Transitions** (one sentence each, so the talk feels like one story):
- 1 → 2: *"With clean data, the next question was: which model predicts best?"*
- 2 → 3: *"But we chose the features and network size by hand. Could a search do better?"*
- 3 → 4: *"The GA showed which features predict. Next we asked: what kinds of patients are there?"*
- 4 → 5: *"Now we have a risk model, a quick model, and patient groups. The agent combines them…"*
- 6 → 7: *"Here's what the agent achieves on patients it has never seen…"*

**Slide charts already available** in [`docs/figures/`](figures/): GA progress, features kept, PCA pictures, LDA,
choosing k, profile disease rates, SOM maps, safety check, plan analysis, app screenshots.
(Part 1 charts are inside notebook 01.)

---

## 4. Cross-training: everyone can cover someone

If someone is absent, or a question comes to the wrong person, the team still needs an answer. With 5 people we use
a **ring**: each person also learns the **next** person's section well enough to present it in an emergency.

```
   Person 1 ──▶ Person 2 ──▶ Person 3 ──▶ Person 4 ──▶ Person 5
      ▲                                                   │
      └───────────────────────────────────────────────────┘
   (Person 1 backs up Person 2, ..., Person 5 backs up Person 1)
```

| Backup | Covers | Why it fits |
|---|---|---|
| Person 1 | Person 2 (models) | Same notebook (01); the models use Person 1's data |
| Person 2 | Person 3 (GA) | The GA trains MLPs and uses the same metrics |
| Person 3 | Person 4 (profiles) | Both find which features matter (GA vs. LDA agree) |
| Person 4 | Person 5 (agent) | The agent uses Person 4's profiles |
| Person 5 | Person 1 (problem, data, results) | The agent's evaluation feeds the final results |

**Everyone** should be able to answer these in one or two sentences:
1. What does the project do? (the one-sentence summary in the overview)
2. Which models are used in the final agent? (two MLPs + k-means, plus search and rules)
3. Why only 73% accuracy? (sick and healthy patients overlap in the data)
4. What's the main novelty? (neuro-symbolic agent: rules stop the model's unsafe advice; search finds the easiest plan)
5. Is it a diagnosis tool? (no: decision support only)

---

## 5. Suggested timeline

| Phase | What happens | Done when |
|---|---|---|
| **Week A: learn** | Everyone reads the overview; each person studies their section's notebook and guide, runs it (Restart → Run All) | You can answer your 5 practice questions without notes |
| **Week B: teach-back** | Team meeting: each person explains their section to the others in 5 minutes; backups quiz the person they cover | Your backup can explain your section back |
| **Week C: build** | Each person makes their slides and writes their paper sections; slides and paper get merged; Person 5 rehearses the demo | Full slide deck and paper draft exist |
| **Week D: rehearse** | Two full run-throughs with timing; practice questions out loud; final fixes | Talk fits the time; demo works on the presentation computer |

---

## 6. Shared jobs

| Job | Owner |
|---|---|
| Merge everyone's slides into one consistent deck | Person 1 |
| Merge paper sections into one document | Person 2 |
| Check every number in the paper and slides matches the notebooks | Person 3 |
| Make sure the code runs on a fresh computer (README setup + troubleshooting) | Person 4 |
| Test the demo on the presentation computer beforehand | Person 5 |
| Final read-through of paper and slides | Everyone |

**Rule for numbers:** every number on a slide or in the paper must come from a notebook output or a file in `results/`.
If two places disagree, check the notebook.
