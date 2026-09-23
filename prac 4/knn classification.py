# K-NEAREST NEIGHBORS (KNN) CLASSIFICATION

# Problem:
# Classify the new point Q = (4,4)

# Training Points:

# P1 = (4,3) -> B
# P2 = (3,3) -> A
# P3 = (5,5) -> A
# P4 = (2,4) -> A
# P5 = (8,8) -> B
# P6 = (7,2) -> B

# Values of K:
# K = 1
# K = 3
# K = 5

# Distance Metric:
# Euclidean Distance


# import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


# create training data

X = np.array([
    [4, 3],    # P1
    [3, 3],    # P2
    [5, 5],    # P3
    [2, 4],    # P4
    [8, 8],    # P5
    [7, 2]     # P6
])

# classes of the training points

y = np.array([
    "B",       # P1
    "A",       # P2
    "A",       # P3
    "A",       # P4
    "B",       # P5
    "B"        # P6
])

# names of the points

point_names = np.array([
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6"
])


# define the new data point

Q = np.array([
    [4, 4]
])


print("\n\t\t")
print("K-NEAREST NEIGHBORS CLASSIFICATION")
print("\t\t")

print("\nNew Point:")
print("Q = (4,4)")


# display training data

training_data = pd.DataFrame({

    "Point": point_names,

    "x1": X[:, 0],

    "x2": X[:, 1],

    "Class": y
})


print("\nTraining Data:")
print(training_data.to_string(index=False))


# calculate euclidean distances

# euclidean Distance:

# d = sqrt((x1 - q1)^2 + (x2 - q2)^2)

# Q = (4,4)

distances = np.sqrt(
    np.sum(
        (X - Q[0]) ** 2,
        axis=1
    )
)


# create distance table

distance_data = pd.DataFrame({

    "Point": point_names,

    "x1": X[:, 0],

    "x2": X[:, 1],

    "Class": y,

    "Distance": distances
})


# sort points from nearest to farthest

distance_data = distance_data.sort_values(
    "Distance"
).reset_index(drop=True)


print("\n\t\t")
print("DISTANCES FROM Q = (4,4)")
print("\t\t")

print(
    distance_data.to_string(
        index=False,
        formatters={
            "Distance":
                lambda x: f"{x:.3f}"
        }
    )
)


# 6. K = 1

# Select the single nearest neighbor.

k = 1


knn_1 = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)


knn_1.fit(
    X,
    y
)


prediction_1 = knn_1.predict(Q)[0]


nearest_1 = distance_data.head(k)


print("\n\t\t")
print("K = 1")
print("\t\t")

print("\nNearest Neighbor:")

print(
    nearest_1[
        ["Point", "Distance", "Class"]
    ].to_string(
        index=False,
        formatters={
            "Distance":
                lambda x: f"{x:.3f}"
        }
    )
)


print(
    "\nPrediction:",
    prediction_1
)

# 7. K = 3

# select the three nearest neighbors.
# the class with the majority vote is selected.

k = 3


knn_3 = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)


knn_3.fit(
    X,
    y
)


prediction_3 = knn_3.predict(Q)[0]


nearest_3 = distance_data.head(k)


print("\n\t\t")
print("K = 3")
print("\t\t")

print("\nThree Nearest Neighbors:")

print(
    nearest_3[
        ["Point", "Distance", "Class"]
    ].to_string(
        index=False,
        formatters={
            "Distance":
                lambda x: f"{x:.3f}"
        }
    )
)


# count class votes

votes_3 = nearest_3["Class"].value_counts()


print("\nClass Votes:")

print(votes_3.to_string())


print(
    "\nPrediction:",
    prediction_3
)


# 8. K = 5

# select the five nearest neighbors.
# the class with the majority vote is selected.

k = 5


knn_5 = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)


knn_5.fit(
    X,
    y
)


prediction_5 = knn_5.predict(Q)[0]


nearest_5 = distance_data.head(k)


print("\n\t\t")
print("K = 5")
print("\t\t")

print("\nFive Nearest Neighbors:")

print(
    nearest_5[
        ["Point", "Distance", "Class"]
    ].to_string(
        index=False,
        formatters={
            "Distance":
                lambda x: f"{x:.3f}"
        }
    )
)


# count class votes

votes_5 = nearest_5["Class"].value_counts()


print("\nClass Votes:")

print(votes_5.to_string())


print(
    "\nPrediction:",
    prediction_5
)

# finall comparison

results = pd.DataFrame({

    "K": [
        1,
        3,
        5
    ],

    "Prediction": [
        prediction_1,
        prediction_3,
        prediction_5
    ]
})


print("\n\t\t")
print("FINAL KNN CLASSIFICATION RESULTS")
print("\t\t")

print(
    results.to_string(
        index=False
    )
)


# comment on the effect of k

print("\n\t\t")
print("OBSERVATION")
print("\t\t")

print(
    """
For K = 1, the prediction depends only on
the nearest point.

For K = 3, the three nearest points participate
in majority voting.

For K = 5, the five nearest points participate
in majority voting.

Therefore, changing K can change the predicted
class because different numbers of neighboring
points are considered.
"""
)


# visualization

plt.figure(
    figsize=(9, 7)
)


# plot class A

class_a = X[y == "A"]

plt.scatter(
    class_a[:, 0],
    class_a[:, 1],
    marker="o",
    s=100,
    label="Class A"
)


# plot class B

class_b = X[y == "B"]

plt.scatter(
    class_b[:, 0],
    class_b[:, 1],
    marker="s",
    s=100,
    label="Class B"
)


# label each training point

for i in range(
    len(X)
):

    plt.annotate(
        point_names[i],
        (
            X[i, 0],
            X[i, 1]
        ),
        xytext=(7, 7),
        textcoords="offset points"
    )


# plot new point Q

plt.scatter(
    Q[:, 0],
    Q[:, 1],
    marker="*",
    s=250,
    label="New Point Q"
)


plt.annotate(
    "Q (4,4)",
    (
        Q[0, 0],
        Q[0, 1]
    ),
    xytext=(10, -15),
    textcoords="offset points"
)


# graph formatting

plt.xlabel(
    "x1"
)

plt.ylabel(
    "x2"
)

plt.title(
    "KNN Classification of Q = (4,4)"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


# save graph

plt.savefig(
    "knn classification.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()