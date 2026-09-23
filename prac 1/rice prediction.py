# RICE PRODUCTION PREDICTION USING MACHINE LEARNING

# LINEAR REGRESSION:
# h(w,b) = w*x + b

# POLYNOMIAL REGRESSION:
# h(w,b) = w2*x^2 + w1*x + b

# MULTIVARIATE LINEAR REGRESSION:
# h(w,b) = w1*x1 + w2*x2 + w3*x3 + b


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# loading data

df = pd.read_csv("rice data.csv")

print("\nOriginal Dataset:")
print(df.head())


# convert faostat data into year wise format

data = df.pivot_table(
    index="Year",
    columns="Element",
    values="Value",
    aggfunc="first"
).reset_index()


# arrange data in chronological order

data = data.sort_values("Year").reset_index(drop=True)


# check dataset

print("\nColumns in Dataset:")
print(data.columns.tolist())

print("\nDataset Size:")
print(data.shape)

print("\nFirst Year:", data["Year"].min())
print("Last Year :", data["Year"].max())


# remove missing values

data = data.dropna(
    subset=[
        "Year",
        "Area harvested",
        "Yield",
        "Production"
    ]
).reset_index(drop=True)


# train-test split

# first 80%  -> training data
# last 20%   -> testing data

# since this is yearly data, we won't shuffle it.

train_size = int(len(data) * 0.80)

train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()


print("\n\t\t")
print("TRAINING AND TESTING DATA")
print("\t\t")

print(
    "Training Period:",
    train["Year"].min(),
    "-",
    train["Year"].max()
)

print(
    "Testing Period :",
    test["Year"].min(),
    "-",
    test["Year"].max()
)

print(
    "Training Observations:",
    len(train)
)

print(
    "Testing Observations:",
    len(test)
)

# target variable

y_train = train["Production"].values
y_test = test["Production"].values


# linear regression

# hypothesis:

# h(w,b) = w*x + b
#
# x = Year
# w = weight / coefficient
# b = bias / intercept

X_train_linear = train[["Year"]]
X_test_linear = test[["Year"]]


linear_model = LinearRegression()

linear_model.fit(
    X_train_linear,
    y_train
)

# get weight and bias

w_linear = linear_model.coef_[0]
b_linear = linear_model.intercept_


# hypothesis function

def h_linear(x, w, b):

    return w * x + b


# test prediction

linear_test_prediction = h_linear(
    test["Year"].values,
    w_linear,
    b_linear
)


print("\n\t\t")
print("LINEAR REGRESSION")
print("\t\t")

print("Hypothesis:")
print("h(w,b) = w*x + b")

print("\nw =", w_linear)
print("b =", b_linear)

print("\nEquation:")

print(
    f"h(w,b) = ({w_linear:.4f})x + "
    f"({b_linear:.4f})"
)


# polynomial regression

# hypothesis:

# h(w,b) = w2*x^2 + w1*x + b

# x = year

# degree = 2
# therefore:

# year -> year and year^2

poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)


# create polynomial features for training data

X_train_poly = poly_features.fit_transform(
    train[["Year"]]
)

# create polynomial features for testing data

X_test_poly = poly_features.transform(
    test[["Year"]]
)


# train polynomial model

poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)


# get weights and bias

w1_poly = poly_model.coef_[0]
w2_poly = poly_model.coef_[1]

b_poly = poly_model.intercept_


# hypothesis function

def h_polynomial(x, w1, w2, b):

    return (
        w2 * x**2
        + w1 * x
        + b
    )


# test prediction

poly_test_prediction = h_polynomial(
    test["Year"].values,
    w1_poly,
    w2_poly,
    b_poly
)


print("\n\t\t")
print("POLYNOMIAL REGRESSION")
print("\t\t")

print("Hypothesis:")
print("h(w,b) = w2*x^2 + w1*x + b")

print("\nw1 =", w1_poly)
print("w2 =", w2_poly)
print("b  =", b_poly)

print("\nEquation:")

print(
    f"h(w,b) = ({w2_poly:.8f})x^2 "
    f"+ ({w1_poly:.4f})x "
    f"+ ({b_poly:.4f})"
)


# multivariate linear regression

# hypothesis:

# h(w,b) = w1*x1 + w2*x2 + w3*x3 + b

# x1 = year
# x2 = area harvested
# x3 = yield

# target = production


X_train_multi = train[
    [
        "Year",
        "Area harvested",
        "Yield"
    ]
]


X_test_multi = test[
    [
        "Year",
        "Area harvested",
        "Yield"
    ]
]


# train multivariate model

multi_model = LinearRegression()

multi_model.fit(
    X_train_multi,
    y_train
)

# get weights and bias

w1_multi = multi_model.coef_[0]
w2_multi = multi_model.coef_[1]
w3_multi = multi_model.coef_[2]

b_multi = multi_model.intercept_


# hypothesis function

def h_multivariate(
    x1,
    x2,
    x3,
    w1,
    w2,
    w3,
    b
):

    return (
        w1 * x1
        + w2 * x2
        + w3 * x3
        + b
    )


# test prediction

multi_test_prediction = h_multivariate(
    test["Year"].values,
    test["Area harvested"].values,
    test["Yield"].values,
    w1_multi,
    w2_multi,
    w3_multi,
    b_multi
)


print("\n\t\t")
print("MULTIVARIATE LINEAR REGRESSION")
print("\t\t")

print("Hypothesis:")
print(
    "h(w,b) = w1*x1 + w2*x2 + w3*x3 + b"
)

print("\nw1 =", w1_multi)
print("w2 =", w2_multi)
print("w3 =", w3_multi)
print("b  =", b_multi)

print("\nEquation:")

print(
    f"h(w,b) = ({w1_multi:.4f})x1 "
    f"+ ({w2_multi:.6f})x2 "
    f"+ ({w3_multi:.4f})x3 "
    f"+ ({b_multi:.4f})"
)


# model evaluation

def evaluate_model(
    model_name,
    actual,
    predicted
):

    # Mean Absolute Error
    mae = mean_absolute_error(
        actual,
        predicted
    )

    # Root Mean Squared Error
    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # R-squared
    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + model_name)
    print("t\t")

    print(
        "MAE  :",
        round(mae, 2)
    )

    print(
        "RMSE :",
        round(rmse, 2)
    )

    print(
        "R2   :",
        round(r2, 4)
    )

    return mae, rmse, r2


# evaluate all three models

linear_scores = evaluate_model(
    "Linear Regression",
    y_test,
    linear_test_prediction
)


poly_scores = evaluate_model(
    "Polynomial Regression",
    y_test,
    poly_test_prediction
)


multi_scores = evaluate_model(
    "Multivariate Regression",
    y_test,
    multi_test_prediction
)


# model comparison table

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],

    "MAE": [
        linear_scores[0],
        poly_scores[0],
        multi_scores[0]
    ],

    "RMSE": [
        linear_scores[1],
        poly_scores[1],
        multi_scores[1]
    ],

    "R2": [
        linear_scores[2],
        poly_scores[2],
        multi_scores[2]
    ]
})


print("\n\t\t")
print("MODEL COMPARISON")
print("\t\t")

print(
    results.round(4).to_string(index=False)
)


# retrain models using all historical data

# we have already tested our models.

# now we use ALL historical data to build
# the final models for future prediction.


# final linear model

final_linear = LinearRegression()

final_linear.fit(
    data[["Year"]],
    data["Production"]
)

w_linear_final = final_linear.coef_[0]
b_linear_final = final_linear.intercept_


# final polynomial model

final_poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)


X_all_poly = final_poly_features.fit_transform(
    data[["Year"]]
)


final_poly = LinearRegression()

final_poly.fit(
    X_all_poly,
    data["Production"]
)


w1_poly_final = final_poly.coef_[0]
w2_poly_final = final_poly.coef_[1]
b_poly_final = final_poly.intercept_


# final multivariate model

final_multi = LinearRegression()

final_multi.fit(
    data[
        [
            "Year",
            "Area harvested",
            "Yield"
        ]
    ],
    data["Production"]
)


w1_multi_final = final_multi.coef_[0]
w2_multi_final = final_multi.coef_[1]
w3_multi_final = final_multi.coef_[2]

b_multi_final = final_multi.intercept_


# future years

future_years = np.array([
    2025,
    2026,
    2027,
    2028,
    2029
])


# predict future area harvested

# the multivariate model requires:

# year
# area harvested
# yield

# but future area harvested is unknown.

# therefore, we estimate future area using
# its historical relationship with year.

area_model = LinearRegression()

area_model.fit(
    data[["Year"]],
    data["Area harvested"]
)


future_area = area_model.predict(
    pd.DataFrame({
        "Year": future_years
    })
)


# predict future yield

# future yield is also unknown.

# therefore, we estimate future yield using
# its historical relationship with year.

yield_model = LinearRegression()

yield_model.fit(
    data[["Year"]],
    data["Yield"]
)


future_yield = yield_model.predict(
    future_years.reshape(-1, 1)
)


# future predictions


# linear regression future prediction

future_linear = h_linear(
    future_years,
    w_linear_final,
    b_linear_final
)


# polynomial regression future prediction

future_poly = h_polynomial(
    future_years,
    w1_poly_final,
    w2_poly_final,
    b_poly_final
)


# multivariate regression future prediction

future_multi = h_multivariate(
    future_years,
    future_area,
    future_yield,
    w1_multi_final,
    w2_multi_final,
    w3_multi_final,
    b_multi_final
)


# create future predicition dataframe

future_results = pd.DataFrame({

    "Year": future_years,

    "Linear Prediction": future_linear,

    "Polynomial Prediction": future_poly,

    "Multivariate Prediction": future_multi
})


# display future predictions

print("\n\n\t\t")
print("FUTURE RICE PRODUCTION PREDICTION")
print("2025 - 2029")
print("\t\t")

print(
    future_results.to_string(
        index=False,
        formatters={

            "Linear Prediction":
                lambda x: f"{x:,.0f}",

            "Polynomial Prediction":
                lambda x: f"{x:,.0f}",

            "Multivariate Prediction":
                lambda x: f"{x:,.0f}"
        }
    )
)


# visualization

# this graph contains:
#
# 1. actual historical production
# 2. linear model test prediction
# 3. polynomial model test prediction
# 4. multivariate model test prediction
# 5. linear future prediction
# 6. polynomial future prediction
# 7. multivariate future prediction

plt.figure(figsize=(14, 8))


# actual historical data

plt.plot(
    data["Year"],
    data["Production"] / 1_000_000,
    marker="o",
    markersize=3,
    label="Actual Production"
)


# linear test prediction

plt.plot(
    test["Year"],
    linear_test_prediction / 1_000_000,
    linestyle="--",
    label="Linear Test Prediction"
)


# polynomial test prediction

plt.plot(
    test["Year"],
    poly_test_prediction / 1_000_000,
    linestyle="--",
    label="Polynomial Test Prediction"
)


# multivariate test prediction

plt.plot(
    test["Year"],
    multi_test_prediction / 1_000_000,
    linestyle="--",
    label="Multivariate Test Prediction"
)


# linear future prediction

plt.plot(
    future_years,
    future_linear / 1_000_000,
    marker="o",
    linestyle=":",
    label="Linear Future Prediction"
)


# polynomial future prediction

plt.plot(
    future_years,
    future_poly / 1_000_000,
    marker="s",
    linestyle=":",
    label="Polynomial Future Prediction"
)


# multivariate future prediction

plt.plot(
    future_years,
    future_multi / 1_000_000,
    marker="^",
    linestyle=":",
    label="Multivariate Future Prediction"
)


# visualization

plt.xlabel(
    "Year",
    fontsize=12
)

plt.ylabel(
    "Rice Production (Million Tonnes)",
    fontsize=12
)

plt.title(
    "Rice Production: Actual, Test Predictions and Future Forecast",
    fontsize=14
)


# train/test boundary

plt.axvline(
    x=test["Year"].min(),
    linestyle="--",
    label="Train/Test Boundary"
)


plt.grid(True)

plt.legend(
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    "graph.png",
    dpi=300,
    bbox_inches="tight")

plt.show()


# save future predictions to CSV file

future_results.to_csv(
    "future predictions.csv",
    index=False
)


print("\n\t\t")
print("prediction file saved successfully!")