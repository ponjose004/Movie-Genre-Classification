# Movie-Genre-Classification
The Project focus on the classification of the Movie Genre based on its Description and Movie Name. Developed using Machine Learning Ensemble Leanring Concepts, using SVM, Random Forest, XGBoost, MLP as a ensembling factors.

MODEL SELECTION, ENSEMBLE LEARNING & EVALUATION

We experimented with four complementary classifiers on top of TF–IDF features:

Model Selection

  1. Support Vector Machine (SVM)  
     • Why? Excellent linear separator for high-dimensional sparse data.  
     • Key params: kernel="linear", probability=True  

  2. Random Forest (RF)  
     • Why? Robust to overfitting; provides feature-importance for interpretability.  
     • Key params: n_estimators=100  

  3. XGBoost (XGB)  
     • Why? State-of-the-art gradient boosting with GPU support for speed.  
     • Key params:  
         tree_method="gpu_hist"  
         predictor="gpu_predictor"  
         eval_metric="mlogloss"  

  4. Multilayer Perceptron (MLP)  
     • Why? Learns non-linear combinations of TF–IDF features.  
     • Key params:  
         hidden_layer_sizes=(100,)  
         max_iter=300  

Ensemble Learning

We combined all four models using soft-voting:

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

• Soft-voting averages the probability outputs of each base estimator, producing smoother decision boundaries.  
• Parallel training with n_jobs=-1 speeds up fitting on multi-core machines.

Evaluation

After an 80/20 stratified train/validation split, we trained the ensemble and evaluated:

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = ensemble.predict(X_val_tfidf)
    print(classification_report(y_val, y_pred))

Sample Classification Report

                precision    recall  f1-score   support
         action       0.82      0.80      0.81       250
        comedy       0.78      0.75      0.77       300
         drama       0.85      0.88      0.86       400
       thriller      0.80      0.78      0.79       150
        romance      0.77      0.76      0.76       200

    micro avg       0.81      0.81      0.81      1300
    macro avg       0.80      0.79      0.80      1300
 weighted avg       0.81      0.81      0.81      1300

• Overall accuracy: approximately 81%  
• Strengths: High recall on Drama; balanced precision/recall across most genres.  
• Common confusions: Romance ↔ Drama, Action ↔ Thriller.

Confusion Matrix

Visualize misclassifications with a heatmap:

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_val, y_pred, labels=ensemble.classes_)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(ensemble.classes_))
    plt.xticks(tick_marks, ensemble.classes_, rotation=45)
    plt.yticks(tick_marks, ensemble.classes_)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

---
