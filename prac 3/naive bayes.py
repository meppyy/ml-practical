# MUSHROOM CLASSIFICATION USING MACHINE LEARNING
# CATEGORICAL NAIVE BAYES

# CLASSIFICATION:
# Predict whether a mushroom is:
# 1. Edible
# 2. Poisonous

# ALGORITHM:
# Categorical Naive Bayes


# import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)


# load the mushroom dataset

df = pd.read_csv(
    "11_mushroom_edibility.csv"
)


print("\nOriginal Dataset:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nDataset Columns:")
print(df.columns.tolist())


# remove SampleID column

# SampleID is only an identifier.
# It does not contain information about mushroom class.
# Therefore, it is removed before training.

df = df.drop(
    columns=["SampleID"]
)


print("\n\t\t")
print("DATASET AFTER REMOVING SampleID")
print("\t\t")

print(df.head())


# check class distribution

print("\n\t\t")
print("CLASS DISTRIBUTION")
print("\t\t")

print(
    df["Class"].value_counts()
)


# separate features (X) and target (y)

# X = Mushroom characteristics
# y = Mushroom class

# Target classes:
# edible
# poisonous

X = df.drop(
    columns=["Class"]
)

y = df["Class"]


print("\n\t\t")
print("FEATURES AND TARGET")
print("\t\t")

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Class")


# label encoding

# the mushroom dataset contains categorical values.

# Example:

# CapShape:
# bell     -> numerical value
# convex   -> numerical value
# flat     -> numerical value
# knobbed  -> numerical value

# CategoricalNB requires categorical features
# represented using numerical category values.


# encode feature columns

feature_encoders = {}


for column in X.columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column]
    )

    feature_encoders[column] = encoder


# encode target class

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    y
)


print("\n\t\t")
print("LABEL ENCODING")
print("\t\t")

print("\nEncoded Features:")

print(
    X.head()
)


print("\nEncoded Target Classes:")

for label, value in zip(
    target_encoder.classes_,
    target_encoder.transform(
        target_encoder.classes_
    )
):

    print(
        label,
        "->",
        value
    )


# train-test split

# 80% -> training data
# 20% -> testing data

# stratify=y ensures that the proportion of
# edible and poisonous mushrooms remains similar
# in both training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n\t\t")
print("TRAINING AND TESTING DATA")
print("\t\t")

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


print("\nTraining Class Distribution:")

print(
    pd.Series(y_train).value_counts()
)


print("\nTesting Class Distribution:")

print(
    pd.Series(y_test).value_counts()
)


# create categorical naive bayes model

# CategoricalNB is designed for categorical features.

# It calculates the probability of each class based
# on the categorical feature values.

naive_bayes_model = CategoricalNB()


# train the model

naive_bayes_model.fit(
    X_train,
    y_train
)


print("\n\t\t")
print("CATEGORICAL NAIVE BAYES MODEL")
print("\t\t")

print(
    "Model trained successfully!"
)


# 10. PREDICT TEST DATA

y_pred = naive_bayes_model.predict(
    X_test
)


# Probability of positive class

# Here:
# 0 = edible
# 1 = poisonous
#
# We take probability of class 1.

y_probability = naive_bayes_model.predict_proba(
    X_test
)[:, 1]


print("\n\t\t")
print("TEST PREDICTIONS")
print("\t\t")

print(
    "First 20 Actual Classes:"
)

print(
    y_test[:20]
)


print(
    "\nFirst 20 Predicted Classes:"
)

print(
    y_pred[:20]
)


# confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n\t\t")
print("CONFUSION MATRIX")
print("\t\t")

print(cm)


# calculate evaluation metrics

# accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


# error Rate

error_rate = 1 - accuracy


# recall / sensitivity

# positive class = poisonous

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)


# precision


precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)

# F1 score

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


# AUC

auc = roc_auc_score(
    y_test,
    y_probability
)


# specificity

# confusion Matrix:

# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()


specificity = tn / (
    tn + fp
)

# display evaluation results

print("\n\t\t")
print("MODEL EVALUATION")
print("\t\t")


print(
    "Accuracy     :",
    round(accuracy, 4)
)


print(
    "Error Rate   :",
    round(error_rate, 4)
)


print(
    "Sensitivity  :",
    round(recall, 4)
)


print(
    "Specificity  :",
    round(specificity, 4)
)


print(
    "Precision    :",
    round(precision, 4)
)


print(
    "F1-Score     :",
    round(f1, 4)
)


print(
    "AUC          :",
    round(auc, 4)
)


# 5-fold stratified cross-validation

# the dataset is divided into 5 folds.

# each fold maintains the proportion of:

# edible
# poisonous

# the model is trained and tested 5 times.

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    naive_bayes_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)


print("\n\t\t")
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("\t\t")


for i, score in enumerate(
    cv_scores,
    start=1
):

    print(
        f"Fold {i} Accuracy:",
        round(score, 4)
    )


print(
    "\nMean CV Accuracy:",
    round(
        cv_scores.mean(),
        4
    )
)


print(
    "Standard Deviation:",
    round(
        cv_scores.std(),
        4
    )
)


# visualize confusion matrix

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm
)


plt.title(
    "Confusion Matrix - Naive Bayes Mushroom Classification"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


# display values inside matrix

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.xticks(
    [0, 1],
    target_encoder.classes_
)


plt.yticks(
    [0, 1],
    target_encoder.classes_
)


plt.colorbar()


plt.tight_layout()


plt.savefig(
    "confusion matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# roc curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve - Naive Bayes Mushroom Classification"
)


plt.legend()


plt.grid(True)


plt.tight_layout()


plt.savefig(
    "roc curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# save model results

results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Error Rate",
        "Sensitivity / Recall",
        "Specificity",
        "Precision",
        "F1-Score",
        "AUC",
        "Mean CV Accuracy"
    ],

    "Value": [
        accuracy,
        error_rate,
        recall,
        specificity,
        precision,
        f1,
        auc,
        cv_scores.mean()
    ]
})


results.to_csv(
    "results.csv",
    index=False
)



print("\n\t\t")



print(
    "Confusion Matrix:"
    "confusion matrix.png"
)


print(
    "ROC Curve:"
    "roc curve.png"
)


print(
    "Model Results:"
    "results.csv"
)


print("\nNaive Bayes practical completed successfully!")