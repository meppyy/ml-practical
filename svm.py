## performing svm on banknote authentication dataset


import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

banknote = fetch_ucirepo(id=267)

X = banknote.data.features
y = banknote.data.targets.iloc[:, 0]

# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. SVM Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# --------------------------------------------------
# 4. Hyperparameter Tuning
# --------------------------------------------------

param_grid = [
    {
        "svm__kernel": ["linear"],
        "svm__C": [0.01, 0.1, 1, 10, 100]
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    },
    {
        "svm__kernel": ["poly"],
        "svm__C": [0.1, 1, 10],
        "svm__degree": [2, 3, 4],
        "svm__gamma": ["scale", 0.01, 0.1]
    }
]

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# --------------------------------------------------
# 5. Best Model
# --------------------------------------------------

best_model = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)

print("\nBest CV Accuracy:")
print(grid.best_score_)

# --------------------------------------------------
# 6. Test Prediction
# --------------------------------------------------

y_pred = best_model.predict(X_test)

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred))

print("\nRecall:")
print(recall_score(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
