# Movie-Genre-Classification
The Project focus on the classification of the Movie Genre based on its Description and Movie Name. Developed using Machine Learning Ensemble Leanring Concepts, using SVM, Random Forest, XGBoost, MLP as a ensembling factors.
```markdown
## 🔍 Model Selection, Ensemble Learning & Evaluation

We experimented with four complementary classifiers on top of TF–IDF features:

1. **Support Vector Machine (SVM)**  
   - **Why?** Excellent linear separator for high-dimensional sparse data.  
   - **Key params:** `kernel="linear"`, `probability=True` for soft-voting.

2. **Random Forest (RF)**  
   - **Why?** Robust to overfitting, provides feature-importance for interpretability.  
   - **Key params:** `n_estimators=100`.

3. **XGBoost (XGB)**  
   - **Why?** State-of-the-art gradient boosting with GPU support for speed.  
   - **Key params:**  
     ```python
     tree_method="gpu_hist"
     predictor="gpu_predictor"
     eval_metric="mlogloss"
     ```

4. **Multilayer Perceptron (MLP)**  
   - **Why?** Learns non-linear combinations of TF-IDF features.  
   - **Key params:**  
     ```python
     hidden_layer_sizes=(100,)
     max_iter=300
     ```

---

## 🏗️ Ensemble Learning

We combine all four models using **soft-voting**:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ("svm", clf_svm),
        ("rf",  clf_rf),
        ("xgb", clf_xgb),
        ("mlp", clf_mlp)
    ],
    voting="soft",    # average predicted probabilities
    n_jobs=-1         # parallelize across cores
)
