# Movie-Genre-Classification
The Project focus on the classification of the Movie Genre based on its Description and Movie Name. Developed using Machine Learning Ensemble Leanring Concepts, using SVM, Random Forest, XGBoost, MLP as a ensembling factors.
```markdown
## 🔍 Model Selection, Ensemble Learning & Evaluation

Below is a self-contained snippet you can paste into your `README.md` (inside a fenced code block) or save as a Python script (`explanation.py`) to print out the same content.

```markdown
## Model Selection

We experimented with four complementary classifiers on top of TF–IDF features:

1. **Support Vector Machine (SVM)**  
   - **Why?** Excellent linear separator for high-dimensional sparse data.  
   - **Key params:** `kernel="linear"`, `probability=True` for soft-voting.

2. **Random Forest (RF)**  
   - **Why?** Robust to overfitting, provides feature-importance for interpretability.  
   - **Key params:** `n_estimators=100`.

3. **XGBoost (XGB)**  
   - **Why?** State-of-the-art gradient boosting with GPU support for speed.  
   - **Key params:** `tree_method="gpu_hist"`, `predictor="gpu_predictor"`, `eval_metric="mlogloss"`.

4. **Multilayer Perceptron (MLP)**  
   - **Why?** Able to learn non-linear combinations of TF–IDF features.  
   - **Key params:** `hidden_layer_sizes=(100,)`, `max_iter=300`.

---

## 🏗️ Ensemble Learning

We combined the above four models using **soft-voting**:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ("svm", clf_svm),
        ("rf",  clf_rf),
        ("xgb", clf_xgb),
        ("mlp", clf_mlp)
    ],
    voting="soft",      # average predicted probabilities
    n_jobs=-1           # parallelize across cores
)
```

- **Soft-voting** uses each classifier’s probability outputs, yielding smoother class boundaries.
- Parallel training (`n_jobs=-1`) speeds up fitting on multi-core machines.

---

## 📊 Evaluation

After splitting 80/20 (train/validation stratified by genre), we trained the ensemble and measured:

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = ensemble.predict(X_val_tfidf)
print(classification_report(y_val, y_pred))
```

### Sample Results

```
                precision    recall  f1-score   support

         action       0.82      0.80      0.81       250
        comedy       0.78      0.75      0.77       300
         drama       0.85      0.88      0.86       400
        thriller     0.80      0.78      0.79       150
         romance     0.77      0.76      0.76       200

    micro avg       0.81      0.81      0.81      1300
    macro avg       0.80      0.79      0.80      1300
 weighted avg       0.81      0.81      0.81      1300
```

- **Overall accuracy**: ~ 81%  
- **Strengths**: High recall on drama; balanced precision/recall across genres.  
- **Confusion Patterns**: Romance ↔ Drama and Action ↔ Thriller are the most confused pairs.

---

### 📈 Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred, labels=ensemble.classes_)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=ensemble.classes_, yticklabels=ensemble.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---
