# 🎬 Movie Genre Classification

<div align="center">

### 🌐 Live Demo

## **[👉 Click Here to Open the App](https://movie-genre-classification-r8g8m8ddgukczxmqt4j4px.streamlit.app/)**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)

</div>

---

## 📌 Project Overview

A **multi-class movie genre classifier** using Natural Language Processing (NLP).  
Given a movie plot description, the model predicts one of **27 genres** including drama, comedy, horror, thriller, sci-fi, western, documentary, and more.

| Attribute | Details |
|---|---|
| **Task** | Multi-class Text Classification (27 genres) |
| **Input** | Movie plot description (free text) |
| **Output** | Predicted genre label |
| **Algorithm** | Ensemble: SVM + Random Forest + XGBoost + MLP |
| **Features** | TF-IDF Vectorizer (10,000 features, unigrams + bigrams) |
| **Deployment** | Streamlit Community Cloud |
| **Language** | Python 3.12 |

---

## 🗂️ Repository Structure

```
Movie-Genre-Classification/
├── app.py                       ← Streamlit web application
├── task1.ipynb                  ← Training notebook (run to generate model)
├── requirements.txt             ← Python dependencies
├── movie_genre_predictions.csv  ← Sample output predictions
├── sample_test.csv              ← Sample CSV for testing the app
├── Data/
│   ├── train_data.txt           ← Training dataset
│   └── test_data.txt            ← Test dataset
└── Model files/
    ├── ensemble_model.pkl       ← Trained ensemble model
    └── tfidf_vectorizer.pkl     ← Fitted TF-IDF vectorizer
```

---

## 🧠 How We Built the Model

### Step 1 — Data Preparation

The raw dataset is in plain text files with this format:

```
ID ::: Movie Title ::: Genre ::: Plot description of the movie...
```

Each line is parsed by splitting on the `' ::: '` delimiter.  
- **Feature (X):** description column  
- **Label (y):** genre column  

The dataset covers **27 unique genres** — drama and documentary are the most frequent; war and biography are the rarest.

---

### Step 2 — Train / Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # preserves genre proportions in both splits
    random_state=42
)
```

> **Why stratify?** Genre distribution is imbalanced — drama has 13,000+ samples while war has only 132. Stratification ensures rare genres appear in both train and val sets proportionally.

---

### Step 3 — TF-IDF Vectorization

Raw text is converted to numerical vectors using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

```
TF-IDF Score = TF(word, document) × IDF(word, corpus)
```

| Component | What it does |
|---|---|
| **TF** (Term Frequency) | How often a word appears in THIS document |
| **IDF** (Inverse Doc Frequency) | Penalizes common words like "the", boosts rare meaningful words |

**Our settings:**

```python
vectorizer = TfidfVectorizer(
    max_features=10_000,    # top 10,000 most informative word/phrase combos
    ngram_range=(1, 2),     # unigrams + bigrams e.g. "serial killer"
    stop_words="english",   # removes filler words: the, is, at...
    lowercase=True          # treats "Drama" and "drama" as same word
)
```

> ⚠️ **Important:** The vectorizer is `fit` ONLY on training data, then used to `transform` both sets — this prevents data leakage.

---

### Step 4 — The Four Models

We train four classifiers, each approaching the TF-IDF matrix differently:

```
┌─────────────────────┐  ┌─────────────────────┐
│        SVM          │  │    Random Forest     │
│  (Linear Kernel)    │  │   (100 trees)        │
│                     │  │                      │
│ Finds the optimal   │  │ Each tree votes on   │
│ hyperplane between  │  │ the genre. Majority  │
│ genre classes in    │  │ rules. Handles noise │
│ high-dim space      │  │ well, rarely overfits│
└─────────────────────┘  └─────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐
│      XGBoost        │  │        MLP           │
│ (Gradient Boosting) │  │ (Neural Network)     │
│                     │  │                      │
│ Learns from errors  │  │ 1 hidden layer of    │
│ of previous trees.  │  │ 100 neurons. Learns  │
│ Highly efficient    │  │ non-linear patterns  │
│ for text features   │  │ via backpropagation  │
└─────────────────────┘  └─────────────────────┘
```

---

### Step 5 — Soft Voting Ensemble

All 4 models are combined using a **Soft Voting Ensemble** — each model outputs a probability for every genre, then probabilities are **averaged** and the highest wins.

```python
ensemble = VotingClassifier(
    estimators=[
        ("svm", clf_svm),
        ("rf",  clf_rf),
        ("xgb", clf_xgb),
        ("mlp", clf_mlp)
    ],
    voting="soft"    # average probabilities, not just votes
)
```

**Example — how soft voting works:**

```
                SVM     RF      XGB     MLP    → AVERAGE
drama:         0.45    0.50    0.48    0.42   →  0.46 ✅ WINNER
comedy:        0.20    0.18    0.22    0.25   →  0.21
horror:        0.15    0.12    0.13    0.16   →  0.14
thriller:      0.10    0.12    0.09    0.10   →  0.10
other:         0.10    0.08    0.08    0.07   →  0.08
```

> Averaging across 4 models makes predictions more robust than relying on any single model alone.

---

### Step 6 — Full Pipeline

```
Raw Text Input
      │
      ▼
Text Preprocessing
(lowercase → remove stopwords → tokenize)
      │
      ▼
TF-IDF Vectorization
(10,000 features | unigrams + bigrams | fit on train only)
      │
      ▼
Train / Val Split
(80% Training  |  20% Validation  |  stratified)
      │
      ▼
┌─────────┬──────────────┬─────────┬─────┐
│   SVM   │ RandomForest │ XGBoost │ MLP │
└─────────┴──────────────┴─────────┴─────┘
      │
      ▼
Soft Voting Ensemble
(average predicted probabilities)
      │
      ▼
Predicted Genre ★
(drama / comedy / horror / thriller / ... 27 classes)
```

---

### Step 7 — Model Evaluation

Evaluated on the 20% validation set:

| Metric | Definition | Performance |
|---|---|---|
| **Accuracy** | % of all correct predictions | ~57% overall |
| **Precision** | Of predicted X, how many are truly X? | High: western (0.87), documentary (0.69) |
| **Recall** | Of actual X, how many did we catch? | High: drama (0.72), documentary (0.83) |
| **F1-Score** | Balance of precision + recall | Best: western (0.80), documentary (0.75) |

> Genres with more training data perform significantly better. Drama, documentary, and comedy are the strongest performers.

---

## 🚀 How to Use the Deployed App

The app has **3 tabs**. Here's exactly how to use each:

---

### 📂 Tab 1 — Upload File (TXT or CSV)

**Option A — Upload CSV:**

1. Download `sample_test.csv` from this repo
2. Open the app → click **"📂 Upload File"** tab
3. Click **"Choose a .txt or .csv file"** → select your CSV
4. Predictions appear instantly in a table
5. Click **"⬇️ Download Predictions CSV"** to save results

Your CSV must have one of these column names:

```
description   OR   plot   OR   summary   OR   overview
```

**Example CSV format:**
```csv
id, title, description
1, Inception, A thief who enters the dreams of others to steal secrets...
2, The Notebook, A poor young man falls in love with a rich young woman...
```

---

**Option B — Upload TXT:**

1. Download `test_data.txt` from the `Data/` folder
2. Upload it in the same file uploader — `.txt` is supported too
3. TXT must follow this format:

```
1 ::: Movie Title ::: Plot description of the movie...
```

---

### ✏️ Tab 2 — Single Description

1. Click the **"✏️ Single Description"** tab
2. Type or paste any movie plot summary into the text box
3. Click **"🔍 Predict Genre"**
4. The predicted genre appears in a green result box

**Try this example:**
```
A young lion prince flees his kingdom after the murder of his father,
only to return years later to reclaim the throne from his evil uncle.
```
Expected output: `ANIMATION` or `DRAMA`

---

### 📊 Tab 3 — Default Test Data

> No upload needed — loads automatically!

1. Click the **"📊 Default Test Data"** tab
2. App auto-loads `Data/test_data.txt` and runs predictions instantly
3. Full table is shown with ID, Title, and Predicted Genre
4. Use the **genre multiselect filter** to view specific genres only
5. A **bar chart** shows the genre distribution of predictions
6. Click **"⬇️ Download Full Predictions CSV"** to export all results

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **scikit-learn** | SVM, Random Forest, MLP, TF-IDF, VotingClassifier |
| **XGBoost** | Gradient boosted trees |
| **Streamlit** | Web app and deployment |
| **pandas** | Data loading and manipulation |
| **joblib** | Model serialization (.pkl files) |
| **Jupyter Notebook** | Model training and experimentation |

---

## 🎭 Supported Genres (27 Classes)

| | | | |
|---|---|---|---|
| drama | comedy | documentary | thriller |
| horror | action | western | sci-fi |
| romance | animation | family | adventure |
| mystery | short | reality-tv | talk-show |
| game-show | sport | music | biography |
| history | news | war | fantasy |
| musical | adult | crime | |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ponjose004/Movie-Genre-Classification.git
cd Movie-Genre-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> To retrain the model, open `task1.ipynb` in Jupyter and run all cells.  
> This will regenerate `Model files/ensemble_model.pkl` and `Model files/tfidf_vectorizer.pkl`.

---

<div align="center">

Built with Python · scikit-learn · XGBoost · Streamlit

[🔗 GitHub Repo](https://github.com/ponjose004/Movie-Genre-Classification) · [🌐 Live App](https://movie-genre-classification-r8g8m8ddgukczxmqt4j4px.streamlit.app/)

</div>
