# MUSHROOM EDIBILITY PREDICTION USING MACHINE LEARNING
# DECISION TREE CLASSIFIER

# CLASSIFICATION:
# Predict whether a mushroom is:
# 1. Edible
# 2. Poisonous

# DECISION TREE:
# Criterion = Entropy

# import necessary libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


# load dataset

df = pd.read_csv(
    "11_mushroom_edibility.csv"
)

print("\nOriginal Dataset:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


# remove SampleID

df = df.drop(
    columns=["SampleID"]
)


print("\nDataset after removing SampleID:")
print(df.head())


# check class distribution

print("\n\t\t")
print("CLASS DISTRIBUTION")
print("\t\t")

print(
    df["Class"].value_counts()
)


# separate features (X) and target (y)


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

# decision tree requires numerical input.

# example:

# edible     -> 0
# poisonous  -> 1

# each categorical feature is also converted
# into numerical values.


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

y = target_encoder.fit_transform(y)


print("\n\t\t")
print("ENCODED DATA")
print("\t\t")

print(
    X.head()
)

print("\nEncoded target classes:")

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

# 80% -> training
# 20% -> testing

# stratify=y ensures that the proportion of
# edible and poisonous mushrooms remains
# approximately the same in both sets.

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
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Training class distribution:"
)

print(
    pd.Series(y_train).value_counts()
)

print(
    "Testing class distribution:"
)

print(
    pd.Series(y_test).value_counts()
)


# create decision tree classifier

# criterion = entropy

# entropy measures the impurity of a node.

# lower entropy means the node is more pure.

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)


# train the decision tree 

decision_tree.fit(
    X_train,
    y_train
)


print("\n\t\t")
print("DECISION TREE TRAINED")
print("\t\t")

print(
    "Criterion:",
    decision_tree.criterion
)

print(
    "Tree Depth:",
    decision_tree.get_depth()
)

print(
    "Number of Leaves:",
    decision_tree.get_n_leaves()
)


# predict test data

y_pred = decision_tree.predict(
    X_test
)


# probability of the positive class
y_probability = decision_tree.predict_proba(
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

accuracy = accuracy_score(
    y_test,
    y_pred
)


error_rate = 1 - accuracy


# Recall / Sensitivity
recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)


# Precision
precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)


# F1 Score
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


# specificity = TN / (TN + FP)

# confusion matrix:

# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()


specificity = tn / (
    tn + fp
)


# display evaluation metrics

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

# each fold maintains the proportion of
# edible and poisonous mushrooms.

# the model is trained 5 times.

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_scores = cross_val_score(
    decision_tree,
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
    "Confusion Matrix - Mushroom Classification"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


# display values inside matrix

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

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

# visualize decision tree

plt.figure(
    figsize=(25, 15)
)


plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=target_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)


plt.title(
    "Decision Tree Classifier - Mushroom Edibility"
)


plt.tight_layout()


plt.savefig(
    "decision tree.png",
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
    "ROC Curve - Mushroom Classification"
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
    "model results.csv",
    index=False
)


print("\n\t\t")
print("OUTPUT FILES SAVED")
print("\t\t")

print(
    "Confusion Matrix:"
    " confusion matrix.png"
)

print(
    "Decision Tree:"
    "decision tree.png"
)

print(
    "ROC Curve:"
    " roc curve.png"
)

print(
    "Model Results:"
    " model results.csv"
)