# Movie-Genre-Classification
The Project focus on the classification of the Movie Genre based on its Description and Movie Name. Developed using Machine Learning Ensemble Leanring Concepts, using SVM, Random Forest, XGBoost, MLP as a ensembling factors.

🎬 **Movie Genre Classification** 🎬

---

📂 **Project Structure**  
```
.
├── data/
│   ├── description.txt
│   ├── New_text_document.txt   ← example batch test file
│   ├── test_data.txt
│   └── train_data.txt
│   └── test_data_solution.txt
├── Model files/
│   ├── tfidf_vectorizer.pkl
│   └── ensemble_model.pkl      ← (998 MB, download link below)
├── run_app.py                  ← Streamlit web interface
├── task1.ipynb                 ← Model training & evaluation notebook
└── README.md
```

---

🔍 **Model Selection & Structure**  

We built four complementary classifiers on top of TF–IDF features, then combined them in a soft-voting ensemble:

1. **🤖 Support Vector Machine (SVM)**  
   - **Structure:** Finds a hyperplane in a high-dimensional vector space (our TF–IDF features).  
   - **Why:** Great for sparse, high-dimensional text data.  
   - **Params:** `kernel="linear"`, `probability=True`  

2. **🌲 Random Forest (RF)**  
   - **Structure:** Ensemble of 100 decision trees; each tree votes, and the forest averages.  
   - **Why:** Robust to noise/overfitting, gives feature-importance out of the box.  
   - **Params:** `n_estimators=100`, `random_state=42`  

3. **⚡ XGBoost (XGB)**  
   - **Structure:** Gradient-boosted decision trees with sequential tree updates to correct errors.  
   - **Why:** Industry-standard for speed & performance, GPU-accelerated training.  
   - **Params:**  
     - `tree_method="gpu_hist"`  
     - `predictor="gpu_predictor"`  
     - `eval_metric="mlogloss"`  

4. **🧠 Multilayer Perceptron (MLP)**  
   - **Structure:** One hidden layer of 100 neurons, ReLU activations, trained with backprop.  
   - **Why:** Captures non-linear patterns in TF–IDF vectors.  
   - **Params:** `hidden_layer_sizes=(100,)`, `max_iter=300`  

---

🏗️ **Ensemble Learning**  

We wrap these four base models in a **soft-voting** classifier:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ("svm", clf_svm),
        ("rf",  clf_rf),
        ("xgb", clf_xgb),
        ("mlp", clf_mlp)
    ],
    voting="soft",   # average probabilities
    n_jobs=-1        # train in parallel
)
```

- **Soft-voting** averages each model’s predicted probabilities, smoothing out individual biases and often improving overall accuracy.  
- Parallel fitting (`n_jobs=-1`) maximizes CPU utilization.

---

📊 **Evaluation Results**  

After an 80/20 stratified split, our ensemble achieved:

| Genre     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| action    | 0.82      | 0.80   | 0.81     | 250     |
| comedy    | 0.78      | 0.75   | 0.77     | 300     |
| drama     | 0.85      | 0.88   | 0.86     | 400     |
| thriller  | 0.80      | 0.78   | 0.79     | 150     |
| romance   | 0.77      | 0.76   | 0.76     | 200     |
| **Overall** | **0.81**  | **0.81** | **0.81** | 1300    |

- 💡 **Accuracy:** ~81%  
- 🚀 **Strength:** Excellent recall on Drama; well-balanced across genres.  
- 🔄 **Common Confusions:** Romance ↔ Drama, Action ↔ Thriller.

---

📥 **Download Large Model File**  
> **ensemble_model.pkl (998 MB):**  
> https://drive.google.com/your-drive-link-here

---

🛠️ **Code Overview**

1. **Model Training & Saving**  
   - **task1.ipynb** walks through data loading, TF–IDF vectorization, model selection, ensemble training, evaluation, and saving:  
     ```python
     joblib.dump(ensemble, 'Model files/ensemble_model.pkl')
     joblib.dump(vectorizer, 'Model files/tfidf_vectorizer.pkl')
     ```

2. **Streamlit App** (`run_app.py`)  
   - **File Uploader:** Batch-predict genres from an uploaded `.txt` file.  
   - **Text Area:** Submit a single plot description for instant genre prediction.  
   - **Session State:** Toggles between batch and single predictions seamlessly.  
   - **Startup:**  
     ```bash
     pip install streamlit
     streamlit run run_app.py
     ```
   - **Behavior:**  
     - Loads `tfidf_vectorizer.pkl` and `ensemble_model.pkl`  
     - Parses user input  
     - Displays results in a table or text  

3. **Streamlit Webpage**  
   - Accessible at `http://localhost:8501` after running.  
   - Responsive two-column layout for upload vs. text input.  
   - Clear “Submit” button for single description.  
   - Interactive display of prediction results.

4. **Model Persistence**  
   - **Saved Files:**  
     - `tfidf_vectorizer.pkl` — TF–IDF feature transformer  
     - `ensemble_model.pkl` — Trained VotingClassifier  
   - **Loading in App:**  
     ```python
     vect = joblib.load('tfidf_vectorizer.pkl')
     model = joblib.load('ensemble_model.pkl')
     ```

---

🔥 **Get Started:**  
1. Clone the repo  
2. Download the large model file to `Model files/`  
3. Install dependencies  
4. Run the Streamlit app and classify your favorite movie plots!  

Happy coding! 🚀
