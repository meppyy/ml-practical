# LOAN APPROVAL PREDICTION USING MACHINE LEARNING
# SUPPORT VECTOR MACHINE (SVM)

# CLASSIFICATION:
# Predict whether a loan application is:

# Y -> Approved
# N -> Not Approved

# ALGORITHM:
# Support Vector Machine

# KERNEL:
# Linear Kernel


# import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


# load dataset

df = pd.read_csv(
    "07_loan_approval.csv"
)

print("\nOriginal Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# remove ApplicantID column

# ApplicantID is only an identifier.
# It does not provide useful information for predicting
# loan approval.

df = df.drop(
    columns=["ApplicantID"]
)


print("\n\t\t")
print("DATASET AFTER REMOVING ApplicantID")
print("\t\t")

print(df.head())


# check class distribution

print("\n\t\t")
print("LOAN APPROVAL CLASS DISTRIBUTION")
print("\t\t")

print(
    df["LoanApproved"].value_counts()
)


# separate features (X) and target (y)

# X = Applicant and loan-related features

# y = LoanApproved

# Y = Approved
# N = Not Approved

X = df.drop(
    columns=["LoanApproved"]
)

y = df["LoanApproved"]


print("\n\t\t")
print("FEATURES AND TARGET")
print("\t\t")

print("\nFeatures:")
print(
    X.columns.tolist()
)

print("\nTarget:")
print(
    "LoanApproved"
)


# label encoding

# SVM requires numerical input.

# The dataset contains categorical features:

# Education
# MaritalStatus
# PropertyArea
# SelfEmployed

# These are converted into numerical values.


# Encode categorical feature columns

categorical_columns = [
    "Education",
    "MaritalStatus",
    "PropertyArea",
    "SelfEmployed"
]


feature_encoders = {}


for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column]
    )

    feature_encoders[column] = encoder


# encode target

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

# stratify=y maintains the proportion of
# Approved and Not Approved applications.

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
    pd.Series(
        y_train
    ).value_counts()
)


print("\nTesting Class Distribution:")

print(
    pd.Series(
        y_test
    ).value_counts()
)


# feature scaling

# SVM is sensitive to feature scale.

# StandardScaler transforms the features so that
# they have approximately:

# Mean = 0
# Standard Deviation = 1

# IMPORTANT:
# the scaler is fitted only on training data.

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print("\n\t\t")
print("FEATURE SCALING")
print("\t\t")

print(
    "Features have been standardized using StandardScaler."
)


# create SVM model

# Kernel = Linear

# C controls the trade-off between:

# 1. maximizing the margin
# 2. allowing classification errors

svm_model = SVC(
    kernel="linear",
    C=1.0,
    probability=True,
    random_state=42
)


# train SVM model

svm_model.fit(
    X_train_scaled,
    y_train
)


print("\n\t\t")
print("SUPPORT VECTOR MACHINE")
print("\t\t")


print(
    "Kernel:",
    svm_model.kernel
)


print(
    "C:",
    svm_model.C
)


print(
    "Number of Support Vectors:",
    svm_model.n_support_
)


print(
    "Total Support Vectors:",
    svm_model.n_support_.sum()
)


# predict on test data

y_pred = svm_model.predict(
    X_test_scaled
)


# probability of positive class

# the encoded target mapping is printed earlier.
# we use the probability corresponding to class 1.

y_probability = svm_model.predict_proba(
    X_test_scaled
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

# confusion matrix:

# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()

specificity = tn / (
    tn + fp
)


# display evaluation metrics

print("\n\t\t")
print("SVM MODEL EVALUATION")
print("\t\t")


print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "Error Rate:",
    round(
        error_rate,
        4
    )
)


print(
    "Sensitivity:",
    round(
        recall,
        4
    )
)


print(
    "Specificity:",
    round(
        specificity,
        4
    )
)


print(
    "Precision:",
    round(
        precision,
        4
    )
)


print(
    "F1-Score:",
    round(
        f1,
        4
    )
)


print(
    "AUC:",
    round(
        auc,
        4
    )
)


# 5-fold stratified cross-validation

# the complete dataset is divided into 5 folds.

# stratification ensures that each fold maintains
# approximately the same proportion of:

# Y -> Approved
# N -> Not Approved


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# scale complete dataset for cross-validation.

# since this practical is focused on implementing SVM,
# the same preprocessing is applied before CV.

X_scaled = scaler.fit_transform(
    X
)


cv_scores = cross_val_score(
    svm_model,
    X_scaled,
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
        round(
            score,
            4
        )
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
    "Confusion Matrix - SVM Loan Approval"
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
    "ROC Curve - SVM Loan Approval"
)


plt.legend()


plt.grid(
    True
)


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
    "svm results.csv",
    index=False
)


# final output

print(
    "Confusion Matrix:"
    "confusion matrix.png"
)


print(
    "ROC Curve:"
    "roc curve.png"
)


print(
    "SVM Results:"
    "svm results.csv"
)


print(
    "\nSVM loan approval practical completed successfully!"
)