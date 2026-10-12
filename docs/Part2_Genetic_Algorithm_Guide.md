# Part 2: A Genetic Algorithm Designs the Model

**Project:** CardioAgent, heart disease risk prediction (CS582 Machine Learning)
**Notebook:** [`02_genetic_algorithm_design.ipynb`](../02_genetic_algorithm_design.ipynb)
**Course link:** Lectures 11–13 (search, genetic algorithms) and Lectures 4–6 (MLP, backpropagation, validation)

> **Part 2 in one sentence:** in Part 1 a human chose which features the model uses and how big the
> neural network is; in Part 2 a **genetic algorithm (GA)** makes both choices automatically by imitating evolution.

---

## Contents

1. [The big idea](#1-the-big-idea)
2. [The files and how they connect](#2-the-files-and-how-they-connect)
3. [The data: how much is used where](#3-the-data-how-much-is-used-where)
4. [Why this is a search problem](#4-why-this-is-a-search-problem)
5. [The notebook, step by step](#5-the-notebook-step-by-step)
6. [The three levels of training and selection](#6-the-three-levels-of-training-and-selection)
7. [Results](#7-results)
8. [What to say in the presentation and paper](#8-what-to-say-in-the-presentation-and-paper)
9. [Glossary](#9-glossary)
10. [Check your understanding](#10-check-your-understanding)
11. [Try it yourself](#11-try-it-yourself)

---

## 1. The big idea

**An analogy: breeding fast dogs.** Start with 20 random dogs. Race them. Let the fastest ones have puppies,
mixing their parents' traits. Now and then a random mutation appears. After 15 generations the dogs are fast.

In Part 2:

| Dog breeding | Our GA |
|---|---|
| A dog | A **model design** (which features + which network shape) |
| Its DNA | A **chromosome**: a string of 13 bits |
| The race | **Fitness**: train the model and measure how well it predicts |
| Fastest dogs have puppies | **Selection**: better designs become parents more often |
| Puppies mix both parents | **Crossover**: children combine parts of two chromosomes |
| Random mutation | **Mutation**: each bit has a 5% chance to flip |

```
   20 random designs
          │
          ▼
   ┌──────────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
   │   FITNESS    │──▶ │ SELECTION │──▶ │ CROSSOVER │──▶ │ MUTATION │──┐
   │ train + score│    │ (roulette)│    │ (one cut) │    │ (5% flip)│  │
   └──────────────┘    └───────────┘    └───────────┘    └──────────┘  │
          ▲                                                            │
          └──────────────────── new generation ◀───────────────────────┘
                                 repeat 15 times
```

---

## 2. The files and how they connect

```
 data/cardio_train.csv            raw data: 70,000 patients
          │ read by
          ▼
 cardio_utils.py                  cleans and prepares the data (same steps as Part 1)
          │ imported by
          ▼
 02_genetic_algorithm_design      runs the GA and evaluates the winner
     .ipynb
          │ saves the winner to
          ▼
 results/ga_best_design.json      best features + shape + threshold + test scores
          │ read later by
          ▼
 Part 4 (the agent)
```

| File | Role | Analogy |
|---|---|---|
| [`cardio_utils.py`](../cardio_utils.py) | Data preparation, written **once** and shared by every notebook | Kitchen prep |
| [`02_genetic_algorithm_design.ipynb`](../02_genetic_algorithm_design.ipynb) | The GA and all the experiments | The cooking |
| `results/ga_best_design.json` | The winning design, so Part 4 doesn't need to re-run the GA | The recipe card |

### What `cardio_utils.py` contains

```
 load_clean_data()                         split_and_scale(df)
 ─────────────────                         ───────────────────
 read the CSV (sep=";")                    stratified split 70 / 15 / 15
   ↓                                         ↓
 remove impossible BP / height / weight    fit the scaler on TRAIN only
   ↓                                         ↓
 add age_years, bmi, is_male               scale train, validation, test
   ↓                                         ↓
 return the clean table (68,559 rows)      return everything in one dictionary
```

It also defines `MODIFIABLE`, the features a patient can change (blood pressure, cholesterol, glucose,
smoking, alcohol, activity, BMI), which the agent will use in Part 4.

**Why a shared file?** If each notebook cleaned the data slightly differently, their results couldn't be compared.
With the same file and the same random seed (42), every notebook uses **exactly the same patients** in each split.

---

## 3. The data: how much is used where

```
                           68,559 patients (after cleaning 70,000)
 ┌───────────────────────────────────────────┬─────────────┬─────────────┐
 │             TRAINING  70%                 │ VALIDATION  │    TEST     │
 │             47,991 patients               │    15%      │    15%      │
 │                                           │   10,284    │   10,284    │
 └───────────────────────────────────────────┴─────────────┴─────────────┘
   learn the weights                           make every    used ONCE for
                                               decision      the final score
```

All three parts keep the same mix (≈ 49.5% sick), because the split is **stratified**.

**Does Part 2 use all the training data?** Yes. Every design is trained on all 47,991 training patients and
scored on all 10,284 validation patients. Nothing is sampled down.

**One detail inside the MLP:** `early_stopping=True` sets aside 10% of the training data internally to decide when to stop:

```
 47,991 training patients
 ┌────────────────────────────────────────────┬────────┐
 │ ~43,192 (90%): weights learned here         │ 4,799  │ → checked after each epoch;
 │                                            │ (10%)  │   training stops when it stops improving
 └────────────────────────────────────────────┴────────┘
```

### Which step uses which data

| Notebook step | Training | Validation | Test |
|---|---|---|---|
| GA fitness (Steps 4 and 6) | ✔ train each MLP | ✔ AUC = fitness | ✘ |
| Random search control (Step 7) | ✔ | ✔ | ✘ |
| Retrain with 5 seeds (Step 8) | ✔ | ✔ compare designs | ✘ |
| Choose threshold (Step 8) | | ✔ | ✘ |
| **Final score** (Step 8) | ✔ final model | | ✔ **only here** |

> **The rule:** TRAINING learns the weights → VALIDATION makes every decision → TEST is touched once, at the end.
> The GA looked at the validation set hundreds of times, so its validation score is slightly optimistic.
> That's why the reported number comes from the test set.

---

## 4. Why this is a search problem

Two decisions have to be made before training:

```
 Decision 1: which features?            Decision 2: which network shape?
 each of 10 features: in or out         one shape from a menu of 8:
                                          (4,)   (8,)    (16,)    (32,)
 2¹⁰ = 1,024 combinations                 (8,4)  (16,8)  (32,16)  (64,32)

                     1,024 × 8 = 8,192 possible designs
```

`(16, 8)` means two hidden layers: 16 neurons, then 8 neurons.

**Why backpropagation can't make these choices:** backprop adjusts *weights*, which are smooth numbers. "Include smoking:
yes or no" is a **jump**, so there's no gradient to follow. Discrete choices like this need **search** (Lecture 11).

**Why not just try all 8,192?** Each design takes about 1 second to train, so all of them would take over 2 hours.
That's possible, but wasteful. With 30 features it would be about 8.6 billion designs, which is impossible.
A GA finds a good design by trying only a few hundred.

---

## 5. The notebook, step by step

### Step 1: Load the data

```python
df = load_clean_data()        # clean table, 68,559 rows
data = split_and_scale(df)    # train 47,991 / validation 10,284 / test 10,284
```

### Step 2: Define the search space

The menu of 8 network shapes is stored in `ARCHITECTURES`. Combined with the 10 features, that gives 8,192 designs.

### Step 3: Encode a design as a chromosome

Every design is written as **13 bits**:

```
            ┌───────────── features: 1 = use, 0 = drop ─────────────┐ ┌ shape ┐
 position:    0     1     2     3     4     5     6     7    8    9    10 11 12
 feature:   ap_hi ap_lo chol  gluc smoke alco  act   age  bmi  male
 bits:        1     0     1     0     0     0     0     1    0    0     1  1  0
                                                                       └──┬──┘
                                                    binary 110 = 6 → ARCHITECTURES[6] = (32, 16)

 decode() → features = [ap_hi, cholesterol, age_years],  shape = (32, 16)
```

This is the actual **winning** chromosome. Three shape bits give 2³ = 8 options, exactly the size of the menu.

### Step 4: The fitness function

```
 chromosome ─▶ decode ─▶ keep only the chosen columns ─▶ train an MLP on TRAINING
                                                                  │
                                                                  ▼
               fitness = AUC on VALIDATION  −  0.002 × number of features
                         └ how well it ranks       └ small penalty, so simpler
                           sick above healthy        designs win ties
```

- **Why AUC and not recall?** A model that says "sick" to everyone has 100% recall but is useless, and
  the GA would find that trick. AUC can't be fooled that way. Recall is tuned later with the threshold.
- **Why the penalty?** A feature must improve AUC by more than 0.002 to earn its place. Fewer features make a
  model simpler, cheaper to collect data for, and easier to explain.
- **Cache:** scores are stored in a dictionary, so a design the GA produces twice is never trained twice.

**Lecture 13's three requirements for a fitness function:**

| Requirement | Ours |
|---|---|
| It must exist | ✔ AUC can be computed for any design |
| Small changes → small score changes | ✔ adding or removing one feature changes AUC a little |
| It must be cheap | ✔ about 1 second per design |

Example: the Part 1 hand-made design (all 10 features, shape (16, 8)) scores **0.7829**.

### Step 5: The three GA operators

```
 ① SELECTION (roulette)       ② CROSSOVER (one cut)                 ③ MUTATION (rate 0.05)
  ┌───────────────┐           A: 1 1 1 1 0 | 0 1 1 1 0 1 0 1          1 0 1 0 0 1 ...
  │  ╱ A ╲   B    │           B: 0 1 0 1 1 | 1 1 1 0 0 0 1 1                ↓ flip
  │ │ 35% │ 25%   │                        ↑ random cut                1 0 1 1 0 1 ...
  │  ╲ C ╱   D    │           child 1: 1 1 1 1 0 | 1 1 1 0 0 0 1 1
  │   20%   20%   │           child 2: 0 1 0 1 1 | 0 1 1 1 0 1 0 1     each bit: 5% chance
  └───────────────┘
  fitter = bigger slice
```

| Operator | What it does | Why it's needed |
|---|---|---|
| **Selection** | Fitter designs become parents more often | Pushes the population toward good designs (**exploitation**) |
| | ...but weak designs keep a small chance | Keeps variety (**exploration**), like Lecture 13's hiring analogy |
| **Crossover** (rate 0.9) | Splices two parents at a random cut | Combines good parts that already exist |
| **Mutation** (rate 0.05) | Flips a few bits at random | Brings in options no parent had. Too much turns the GA into random search |

**One practical detail about the roulette wheel:** all fitness values are close together (≈ 0.77–0.79), so a plain wheel
would give everyone almost the same slice. We subtract the worst score first, so the slices reflect how much **better**
each design is than the worst.

### Step 6: Run the GA

| Setting | Value | Meaning |
|---|---|---|
| Population | 20 | designs per generation |
| Generations | 15 | the stopping rule |
| Elitism | 2 | the 2 best designs are copied unchanged, so the best is never lost |
| Crossover rate | 0.9 | |
| Mutation rate | 0.05 | per bit |

```
 Generation 0: 20 random chromosomes
        │
        ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. score all 20 with fitness()                         │
 │ 2. copy the best 2 unchanged             (elitism)     │
 │ 3. fill the other 18 slots:                            │
 │      select 2 parents → crossover → mutate → 2 children│
 │ 4. that is the new generation                          │
 └──────────────────────────┬─────────────────────────────┘
                            │ × 15
                            ▼
                     best chromosome
```

### Step 7: Control experiment (is the GA smart, or lucky?)

Random search got the **same budget**: 191 random designs. If it did just as well, the GA wouldn't be adding anything.

### Step 8: GA design vs. hand-made design, and the final test

1. Retrain both designs with **5 different random seeds**, so a lucky starting point can't mislead us.
2. Choose each design's **threshold on validation**: the highest one that still catches ≥ 80% of sick patients.
3. Score both **once on the test set**.

### Step 9: Save the winner

Write the design to `results/ga_best_design.json` for the agent in Part 4.

---

## 6. The three levels of training and selection

Model selection happens at three nested levels:

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │ LEVEL 3: FINAL CHECK     5 seeds · threshold on validation · test ONCE│
 │  ┌─────────────────────────────────────────────────────────────────┐  │
 │  │ LEVEL 2: GA DESIGN SEARCH   which features? which shape?        │  │
 │  │          191 designs tried, scored on validation                │  │
 │  │  ┌───────────────────────────────────────────────────────────┐  │  │
 │  │  │ LEVEL 1: TRAIN ONE MLP   learn the weights by backprop    │  │  │
 │  │  └───────────────────────────────────────────────────────────┘  │  │
 │  └─────────────────────────────────────────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────────┘
```

### Level 1: Training one MLP (learning the weights)

For the winning design (3 inputs → 32 → 16 → 1 output) there are about **673 numbers** to learn:

```
  INPUT (3)        HIDDEN 1 (32)       HIDDEN 2 (16)       OUTPUT (1)
  ap_hi  ○─┐       ○                   ○
  chol   ○─┼─────▶ ○  ...  ──────────▶ ○  ...  ──────────▶  ○  P(disease)
  age    ○─┘       ○                   ○

      3×32 + 32 = 128        32×16 + 16 = 528         16×1 + 1 = 17     → 673 weights and biases
```

```
 START: weights = small random numbers (random_state decides which)
   │
   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ ONE EPOCH = one pass through ~43,192 training patients           │
 │   shuffle, cut into mini-batches of 200 (~216 batches)           │
 │   for each batch:                                                │
 │     1. FORWARD   200 patients → 200 predictions                  │
 │     2. LOSS      how wrong? (one number)                         │
 │     3. BACKWARD  backpropagation: each weight's share of blame   │
 │     4. UPDATE    nudge all weights to reduce the error           │
 │   after the epoch: check the 4,799 held-out patients             │
 └─────────────────────────────┬────────────────────────────────────┘
          improved? ── yes ────┴──▶ next epoch
             │
             no, for 10 epochs in a row
             ▼
 STOP → keep the weights from the best epoch       (early stopping, Lecture 6)
```

**A single patient (illustrative numbers):** `ap_hi = 150, chol = 3, age = 58` is scaled to `[+1.4, +2.4, +0.6]`; truth = 1 (sick).
The forward pass predicts 0.62, so loss = −log(0.62) = 0.48. Backprop assigns blame, the weights are nudged, and next time a
similar patient might get 0.66.

**Differences from the lectures** (same idea, practical upgrades):

| | Lectures 3–5 | scikit-learn here |
|---|---|---|
| Hidden activation | Sigmoid | **ReLU**: trains faster in hidden layers (Lecture 16) |
| Weight update | Plain gradient descent | **Adam**: gradient descent with an automatic step size per weight |
| Update style | Batch or sequential | **Mini-batch** of 200, between the two |

**Why early stopping?** Training loss keeps falling, but loss on unseen patients eventually rises as the model starts
memorising. Stopping at the best point prevents overfitting.

### Level 2: The GA chooses the design

Level 1 learns **weights** for a fixed design. Features and shape are **hyperparameters**: settings chosen before
training. The GA chooses them using **validation** AUC, because a model always looks good on the patients it learned from.

| Design | Fitness |
|---|---|
| Part 1: 10 features, (16, 8) | 0.7829 |
| Gen 0 best: 7 features, (16, 8) | 0.7885 |
| Gen 4 best: 5 features, (64, 32) | 0.7912 |
| **Gen 11 best: 3 features, (32, 16)** | **0.7938** ◀ winner |

### Level 3: Confirming the winner

1. **Lucky or real?** 5 seeds and a random-search control (results below).
2. **Threshold:** 0.50 would catch only ~71% of sick patients, so **0.36** is chosen on validation to reach ≥ 80%.
3. **Test once:** the honest final score.

### Who decides what

| What is decided | Who decides | Data used |
|---|---|---|
| The ~673 **weights** | Backpropagation (Level 1) | Training (~43,192) |
| **When to stop** training | Early stopping | 10% held out from training (4,799) |
| **Features + shape** | The GA (Level 2) | Validation (10,284) |
| **Threshold** (0.36) | Recall ≥ 80% rule (Level 3) | Validation (10,284) |
| **Final reported score** | Nobody, it's only measured | Test (10,284), once |

---

## 7. Results

### GA progress

![GA progress over 15 generations](figures/part2_ga_progress.png)

- **Blue (best):** never goes down, thanks to elitism. It beat the Part 1 design (dashed line) from generation 0.
- **Orange (average):** rises fast as good genes spread. The dips (generations 7 and 14) are mutation trying new,
  sometimes worse, designs. That's exploration at work.

```
 Gen  0 | best 0.7885 | 7 features, (16, 8)     ← random start
 Gen  1 | best 0.7903 | 5 features, (32,)
 Gen  4 | best 0.7912 | 5 features, (64, 32)
 Gen 10 | best 0.7922 | 4 features, (64, 32)
 Gen 11 | best 0.7938 | 3 features, (32, 16)    ← winner
```

The GA trained **191 of the 8,192** possible designs in about 2 minutes.

### Which features survived

![Fraction of the final population using each feature](figures/part2_features_kept.png)

`age_years` and `ap_hi` are in 95% of the final population, then `cholesterol` (65%). `smoke` is in only 5%.
This agrees with Part 1's correlation chart and Random Forest importance: **three different methods agree.**

### The winning design

| | |
|---|---|
| Features | `ap_hi` (systolic BP), `cholesterol`, `age_years` |
| Dropped | `ap_lo`, `gluc`, `smoke`, `alco`, `active`, `bmi`, `is_male` |
| Shape | (32, 16) |
| Validation fitness | 0.7938 |
| Threshold | 0.36 |

### Is the GA better than random search? (same budget: 191 designs)

| Method | Best fitness |
|---|---|
| **GA** | **0.7938** |
| Random search | 0.7903 |

### GA design vs. hand-made design (validation AUC, 5 seeds)

| Design | Features | Mean AUC | Spread (std) |
|---|---|---|---|
| Part 1, hand-made, (16, 8) | 10 | 0.8009 | 0.0015 |
| **GA design, (32, 16)** | **3** | **0.7995** | **0.0002** |

The difference (≈ 0.001) is about the size of the seed-to-seed spread: **the same performance with 70% fewer inputs.**

### Final test set (used once)

| Design | Features | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Part 1, hand-made | 10 | 0.41 | 0.716 | 0.685 | 0.789 | 0.733 | 0.794 |
| **GA design** | **3** | **0.36** | **0.710** | **0.677** | **0.790** | **0.729** | **0.789** |

Both catch about **79% of sick patients**.

### The final selected model

```
 ┌──────────────────────────────────────────────┐
 │ MLP, shape (32, 16), ReLU, Adam              │
 │ inputs: ap_hi, cholesterol, age_years        │  ← chosen by the GA
 │ weights: learned by backprop on TRAINING     │  ← Level 1
 │ threshold: 0.36                              │  ← chosen on VALIDATION
 │ honest performance: recall 79%, AUC 0.789    │  ← measured on TEST
 └──────────────────────────────────────────────┘
```

### Key insight and what it means for the agent

Almost all the predictive information in this dataset is in **blood pressure, cholesterol and age**. Lifestyle features
(smoking, alcohol, activity) add almost nothing *to prediction*, probably because they are self-reported.

That creates a design decision for Part 4: the 3-feature model is ideal for quick **screening**, but it can't answer
"what if I quit smoking?" because it never sees `smoke`. So the agent will use **the GA model to predict** and
**a full-feature model for what-if scenarios**.

---

## 8. What to say in the presentation and paper

1. **Problem:** choosing features and network shape is a *discrete search* problem (8,192 designs); backprop can't solve it.
2. **Method:** a GA with a 13-bit chromosome, fitness = validation AUC − 0.002 × number of features, roulette selection,
   one-point crossover (0.9), mutation (0.05), and elitism (2), over 15 generations of 20.
3. **Result:** 3 features match the accuracy of all 10 (AUC 0.800 vs 0.801); test recall is 79%.
4. **Rigour:** compared with random search on the same budget and repeated over 5 seeds; the test set was used once.
5. **Insight:** the GA, correlation and Random Forest importance all point to blood pressure, cholesterol and age.
6. **Honest limitations:**
   - The GA's edge over random search is small (0.794 vs 0.790), which suggests many designs are nearly as good.
   - Only one GA run is reported. Repeating it with several seeds would strengthen the claim.
   - Fitness uses a single validation split. Cross-validation would be more robust, but slower.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **Feature** | An input column, e.g. `ap_hi` |
| **Hyperparameter** | A setting chosen *before* training, e.g. which features to use or the network shape |
| **Weight** | A number inside the network, learned *during* training |
| **Chromosome** | A design written as a string of bits (13 here) |
| **Gene / bit** | One position in the chromosome |
| **Population** | The set of chromosomes in one generation (20 here) |
| **Generation** | One round of score → select → crossover → mutate |
| **Fitness** | How good a chromosome is (validation AUC − penalty) |
| **Roulette selection** | Pick parents at random, with fitter ones more likely |
| **Crossover** | Cut two parents and swap their tails |
| **Mutation** | Randomly flip a bit (5% chance per bit) |
| **Elitism** | Copy the best chromosomes unchanged into the next generation |
| **Exploitation / exploration** | Using what already works / trying new things |
| **Epoch** | One pass of the MLP through all its training data |
| **Early stopping** | Stop training when held-out performance stops improving |
| **ROC-AUC** | How well a model ranks sick above healthy (0.5 = coin flip, 1.0 = perfect) |
| **Recall** | Fraction of truly sick patients the model catches |
| **Threshold** | Probability above which we predict "sick" (0.36 here) |
| **Random seed** | A number that fixes the randomness so results can be reproduced |

---

## 10. Check your understanding

<details>
<summary>1. Why can't backpropagation choose which features to use?</summary>

Backprop follows gradients of smooth numbers (weights). Including or excluding a feature is a yes/no jump with no
gradient, so it needs search.
</details>

<details>
<summary>2. What does the chromosome <code>[1,1,0,0,0,0,0,1,1,0, 1,0,1]</code> mean?</summary>

Features `ap_hi`, `ap_lo`, `age_years`, `bmi`. Shape bits 101 = 5 → `(16, 8)`.
</details>

<details>
<summary>3. Why is the fitness AUC and not recall?</summary>

Predicting "sick" for everyone gives 100% recall, and the GA would find that useless trick. AUC rewards correct ranking.
</details>

<details>
<summary>4. Why do weak designs still get selected sometimes?</summary>

To keep variety (exploration). Always breeding only the top two leads to getting stuck in a local optimum.
</details>

<details>
<summary>5. Why is the GA scored on validation data, not training data?</summary>

A model always looks good on the patients it learned from. Validation measures how well a design generalizes to new patients.
</details>

<details>
<summary>6. Why isn't the GA's validation score the number we report?</summary>

The GA chose the design by looking at validation hundreds of times, so that score is slightly optimistic.
The test set played no part in any choice, so it gives the honest score.
</details>

<details>
<summary>7. Why retrain with 5 seeds?</summary>

Training starts from random weights. One run might be lucky; the mean and spread over 5 runs show whether a difference is real.
</details>

<details>
<summary>8. Why does the agent need two models?</summary>

The 3-feature GA model is best for prediction, but it never sees lifestyle features, so it can't answer "what if I quit
smoking?". A full-feature model handles what-if questions.
</details>

---

## 11. Try it yourself

Change one setting in the notebook, re-run it, and compare with the results above:

| Change | Question to answer |
|---|---|
| `FEATURE_PENALTY = 0` | How many features does the GA keep when extra features are free? |
| `FEATURE_PENALTY = 0.01` | Does it drop down to 1–2 features? How much AUC is lost? |
| `MUTATION_RATE = 0.2` | Does the GA start behaving like random search? |
| `POPULATION_SIZE = 40` | Does a bigger population find a better design? How much longer does it take? |
| Add `(32, 16, 8)` to `ARCHITECTURES` | Remember: 9+ shapes need 4 shape bits (`N_SHAPE_BITS = 4`) |

**Next:** Part 3, patient profiles with k-means / SOM (Lectures 9–10) and PCA / LDA plots (Lecture 7).
