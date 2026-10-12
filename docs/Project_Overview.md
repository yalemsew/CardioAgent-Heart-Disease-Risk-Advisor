# CardioAgent: Project Overview in Plain Language

*A beginner-friendly walkthrough of the whole project, from beginning to end. No machine-learning background needed.
Each part has a detailed study guide, linked below.*

---

## The project in one sentence

> **We built a computer assistant that looks at a person's routine checkup results, estimates their chance of having heart disease,
> explains why, and suggests the easiest lifestyle changes that would lower that chance.**

---

## Chapter 1: The problem

Heart disease is one of the biggest causes of death in the world. Doctors already measure things like blood pressure and
cholesterol at routine checkups. **Can a computer use those everyday numbers to spot people at risk early?**

Our assistant doesn't replace a doctor. It's a **helper**, like a smoke alarm that tells you to look more closely, not a firefighter.

---

## Chapter 2: The data

We used a public dataset of **70,000 real patients**. For each person we know:

```
 age · sex · height · weight · blood pressure · cholesterol · blood sugar
 smokes? · drinks alcohol? · physically active?
                          +
 THE ANSWER: does this person have heart disease? (yes / no)
```

About half the patients have heart disease and half don't.

**Cleaning:** some records contained impossible values, such as a blood pressure of −150 or 16,020 (typing mistakes).
We removed about 2% of patients like that, leaving **68,559**.

**Splitting:** we divided the patients into three groups, like a student preparing for an exam:

```
 70%  TRAINING    → the computer studies these patients (the textbook)
 15%  VALIDATION  → used for practice tests while making choices
 15%  TEST        → the final exam: kept hidden until the very end
```

The final exam must contain patients the computer **has never seen**. Otherwise it could just memorize the answers.

---

## Chapter 3 (Part 1): Teaching the computer to predict

*Notebook: [`01_cardio_basics.ipynb`](../01_cardio_basics.ipynb)*

We used a **neural network**, a program loosely inspired by the brain (Lectures 4–5). It works like this:

```
 1. Look at a patient            → guess "62% chance of heart disease"
 2. Check the real answer        → "this patient was sick, the guess was too low"
 3. Adjust a little              → next time, guess higher for similar patients
 4. Repeat for 48,000 patients, many times over
```

After training, it can make a guess for **new** patients.

**How good is it?** It's right about **73% of the time**. We also tried three other methods from class (SVM, Random Forest,
Naive Bayes), and **they all landed around 73% too**. That told us the limit comes from **the data itself**: many sick and
healthy people simply have very similar checkup numbers.

**One important choice:** in medicine, **missing a sick person is worse than a false alarm**. So we tuned the assistant to be a
bit "cautious": it now **catches about 79% of sick people**, at the cost of a few more false alarms.

---

## Chapter 4 (Part 2): Letting evolution design the model

*Notebook: [`02_genetic_algorithm_design.ipynb`](../02_genetic_algorithm_design.ipynb) · Guide: [`Part2_Genetic_Algorithm_Guide.md`](Part2_Genetic_Algorithm_Guide.md)*

Before training, someone has to decide two things:
- **Which patient information** should the model look at? (all 10 items, or just some?)
- **How big** should the neural network be?

There are **8,192 possible combinations**. Instead of guessing, we used a **genetic algorithm** (Lectures 12–13), which copies evolution:

```
 start with 20 random designs
   → test each one (how well does it predict?)
   → the best ones "have children" (mix their choices)
   → small random changes appear (mutations)
   → repeat 15 generations
```

**The discovery:** only **3 pieces of information** are needed: **blood pressure, cholesterol and age**. A model using just these 3 is
**as accurate as one using all 10**. The other items, like smoking and alcohol, added almost nothing in this dataset.

That's useful: a nurse can do a **quick screening** with only 3 numbers.

---

## Chapter 5 (Part 3): Finding types of patients

*Notebook: [`03_patient_profiles.ipynb`](../03_patient_profiles.ipynb) · Guide: [`Part3_Patient_Profiles_Guide.md`](Part3_Patient_Profiles_Guide.md)*

Next we asked a different question: **"What kinds of patients are there?"**

This time we **hid the answers** and let the computer sort patients into groups of similar people by itself (Lectures 9–10).
It's like sorting a pile of photos into piles without being told the categories.

It found **4 types of patients**:

| Group | What they have in common | Have heart disease |
|---|---|---|
| 🔴 High blood pressure & overweight | high BP, heavy, older | **79%** |
| 🟠 High cholesterol & sugar | very high cholesterol and blood sugar | **62%** |
| 🟡 Smokers & drinkers | mostly men who smoke | **46%** |
| 🟢 Lower-risk | normal values, younger | **33%** |

The computer never saw who was sick, yet its groups turned out to have very different disease rates. Two completely different
methods (k-means and SOM) found **the same 4 groups**, so we trust them.

These groups help **explain** a prediction: "You're in the high blood pressure group, where 79% of people have heart disease"
is easier to understand than a bare number.

---

## Chapter 6 (Part 4): Building the assistant (the "agent")

*Notebook: [`04_agent.ipynb`](../04_agent.ipynb) · Code: [`cardio_agent.py`](../cardio_agent.py) · Guide: [`Part4_Agent_Guide.md`](Part4_Agent_Guide.md)*

Now we put everything together into one assistant. Like every **agent** from Lecture 11, it works in three steps:

```
 1. PERCEIVE   check the numbers make sense
               (e.g. reject blood pressure 80/120 — the top number must be higher)

 2. THINK      • estimate the risk                  (Part 1's model)
               • find the patient's group           (Part 3)
               • search for the EASIEST changes     (Lecture 12)
                 that bring the risk down

 3. ACT        write a clear report
```

**The search** works like route planning on a map. Each "move" is one change (lower blood pressure a bit, lower cholesterol a level,
lose some weight, become active), and each move has an effort cost. The assistant finds the **cheapest route** to a safe risk level.

### The big surprise

When we asked the model *"what happens if a smoker quits?"*, it said **the risk goes up**. That's obviously wrong. It happened because
the model learned **patterns, not causes**: in this dataset the smokers were mostly younger men, who had less heart disease overall.

An assistant that blindly trusted the model would effectively tell people to **keep smoking**.

**Our fix:** we gave the assistant a small **rule book** of medical facts:
- It may only suggest **healthy** changes.
- Advice on smoking, alcohol and sugar comes **from medical guidelines**, not from the model.
- It never displays a healthy change as harmful.

Combining a learned model with hand-written rules is called **neuro-symbolic AI**, and this became the most important original idea
in the project.

---

## Chapter 7 (Part 5): A simple web page

*Code: [`app.py`](../app.py) · Guide: [`Part5_Demo_App_Guide.md`](Part5_Demo_App_Guide.md)*

So that anyone can use the assistant without writing code, we built a **web page**. You type the values into a form, and the
report appears instantly:

```
 Risk 88% (HIGH) · group: "smokers & drinkers"
 Main factors: blood pressure, cholesterol, weight
 Easiest plan: lower BP to 125/80 + cholesterol to normal + become active → about 36%
 Also: stop smoking (medical advice)
 This is not a diagnosis. Talk to a doctor.
```

To start it, run `streamlit run app.py` from inside the `cardio_project` folder, then open http://localhost:8501 in a browser.
(See the README for setup and troubleshooting.)

---

## Chapter 8: The final results

All measured on the **hidden test patients**:

| Question | Answer |
|---|---|
| How many sick people does it catch? | **About 79%** |
| Does the quick 3-number screening work? | **Yes**, also about 79% |
| Can it find a plan that lowers risk to a safe level? | **For 66%** of at-risk patients: always for under-55s, rarely for over-60s (age can't be changed) |
| How much do the plans help on average? | Risk goes from **71% → 39%** |
| Did it ever suggest something unhealthy? | **Never** (0 of 1,000 plans) |

---

## Chapter 9: Honest limitations

- **73% accuracy is a ceiling set by the data.** Basic checkup numbers can't tell everyone apart.
- **The what-if numbers are patterns, not proof.** "Lowering blood pressure would reduce your risk by X" is the model's estimate,
  not a medical guarantee.
- **Important information is missing,** such as family history, ECG results and medication.
- **No doctor has tested it.** It's a student project and a decision-support tool, not a medical device.

---

## The whole project in one picture

```
 70,000 patients
       │
       ▼
 clean + split  ─────────────────────────────────────────┐
       │                                                 │
       ├──▶ Part 1: neural network learns to predict     │
       │           (catches ~79% of sick people)         │
       │                                                 │
       ├──▶ Part 2: evolution finds the 3 key numbers    │
       │           (BP, cholesterol, age)                │
       │                                                 │
       ├──▶ Part 3: finds 4 types of patients            │
       │           (without seeing who is sick)          │
       │                                                 │
       └──▶ Part 4: THE ASSISTANT ◀──────────────────────┘
                 perceive → think (+ rule book) → act
                       │
                       ▼
            Part 5: web page anyone can use
```

---

## Which course topics it uses

| Course topic | Where it appears |
|---|---|
| Neural networks, backpropagation (L3–5) | Part 1: the prediction model |
| Validation, overfitting (L6) | The train/validation/test split, early stopping |
| PCA, LDA (L7) | Part 3: pictures of the data |
| SVM (L8), Random Forest, Bayes (L15) | Part 1: comparison models |
| K-means (L9), SOM (L10) | Part 3: patient groups |
| Agents, PEAS, search (L11–12) | Part 4: the assistant and its planning |
| Genetic algorithms (L12–13) | Part 2: designing the model |

---

## Project status

```
 ✅ Parts 1–5: all the machine learning, the assistant, and the web page
 ⬜ Part 6:   the paper and presentation slides
```

## Where to go next

| If you want to… | Read |
|---|---|
| Set up and run the project | [`README.md`](../README.md), sections *How to run* and *Troubleshooting* |
| Understand the genetic algorithm | [`Part2_Genetic_Algorithm_Guide.md`](Part2_Genetic_Algorithm_Guide.md) |
| Understand the patient groups | [`Part3_Patient_Profiles_Guide.md`](Part3_Patient_Profiles_Guide.md) |
| Understand the assistant and its safety rules | [`Part4_Agent_Guide.md`](Part4_Agent_Guide.md) |
| Use or demo the web page | [`Part5_Demo_App_Guide.md`](Part5_Demo_App_Guide.md) |

Each guide starts with an **"In simple words"** section.
