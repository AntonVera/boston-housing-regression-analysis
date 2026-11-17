# Boston Housing Regression Analysis
# Author: Antonio Vera
# Academic coursework – FAU Erlangen-Nürnberg

import pandas as pd  # for tables
import numpy as np  # for calculations
import matplotlib.pyplot as plt  # for plots

from sklearn.linear_model import LinearRegression  # Linear Regression as a regression model
from sklearn.preprocessing import PolynomialFeatures  # to model polynomial relationships, e.g. y = c + m1x + m2x² + m3x³
from sklearn.model_selection import KFold, cross_val_score  # to compare different regression degrees
from sklearn.pipeline import make_pipeline  # to apply PolynomialFeatures and LinearRegression together
from sklearn.metrics import r2_score, mean_squared_error  # model evaluation metrics


# Questions:
# Are these relationships linear?
# Which regression degree best describes each relationship?
# Do property prices fall as crime rates rise?
# How much does the property price increase the closer the property is to the city center?
# Do better educational opportunities also affect house prices?


csv_file = "boston_housing.csv"
df = pd.read_csv(csv_file)


print("First Data lines:")
print(df.head())


print("\nColumns:")
print(df.columns.tolist())


print("\nStatistics:")
print(df.describe())


target = "house_price"  # target variable

features = [
    "crime_rate",
    "distance_to_center",
    "students_per_teacher"
]  # input variables

degrees = [1, 2, 3, 4, 5]  # polynomial degrees - Degree 1: simple linear regression

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)  # we use KFold to compare the models more robustly

results = []  # list used to store the results


for feature in features:

    X = df[[feature]].values  # X contains one column as the input variable
    y = df[target].values  # y contains the house price as the target variable

    correlation = np.corrcoef(
        df[feature],
        y
    )[0, 1]  # correlation between the input variable and the house price


    for degree in degrees:

        # Create the model:
        # PolynomialFeatures generates different polynomial degrees:
        # x, x², ..., xⁿ
        model = make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            ),
            LinearRegression()
        )


        # Calculate the mean R² score using 5-fold cross-validation.
        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2"
        )


        # Train the model using all data to obtain the regression curve for the plot.
        model.fit(X, y)


        # Calculate predictions on the training data.
        y_pred = model.predict(X)


        # R² on the training data.
        train_r2 = r2_score(y, y_pred)


        # RMSE: average prediction error in the same units as the house price.
        rmse = np.sqrt(
            mean_squared_error(y, y_pred)
        )


        results.append({
            "feature": feature,
            "degree": degree,
            "correlation": correlation,
            "cv_r2_mean": cv_scores.mean(),  # mean
            "cv_r2_std": cv_scores.std(),  # standard deviation
            "train_r2": train_r2,
            "rmse": rmse
        })


# Convert the results list into a DataFrame.
results_df = pd.DataFrame(results)


# Save the results table as a CSV file.
results_df.to_csv(
    "regression_results.csv",
    index=False
)


# Display the rounded results table.
print("\nRegression results:")
print(results_df.round(4))


# Determine the best polynomial degree for each variable
# based on the mean cross-validation R² score.
best_models = results_df.loc[
    results_df.groupby("feature")["cv_r2_mean"].idxmax()
]


# Display the best models.
print("\nBest model after Cross-Validation-R²:")
print(best_models.round(4))


# Plots
# Plot each variable

for feature in features:

    X = df[[feature]].values  # input values
    y = df[target].values  # house prices


    # Read the best polynomial degree for this variable
    # from the results table.
    best_degree = int(
        best_models[
            best_models["feature"] == feature
        ]["degree"].iloc[0]
    )


    # Create the best model again.
    best_model = make_pipeline(
        PolynomialFeatures(
            degree=best_degree,
            include_bias=False
        ),
        LinearRegression()
    )


    # Train the model using all available data.
    best_model.fit(X, y)


    # Generate ordered x-values for a smooth regression curve.
    x_grid = np.linspace(
        X.min(),
        X.max(),
        300
    ).reshape(-1, 1)


    # Calculate model predictions for the generated x-values.
    y_grid = best_model.predict(x_grid)


    # Create a new plot.
    plt.figure(figsize=(8, 5))


    # Plot the real data points.
    plt.scatter(
        X,
        y,
        alpha=0.6,
        label="Data points"
    )


    # Plot the regression curve.
    plt.plot(
        x_grid,
        y_grid,
        linewidth=2,
        label=f"Regression Degree {best_degree}"
    )


    # Set the plot title.
    plt.title(
        f"House Price vs. {feature}"
    )


    # Set the x-axis label.
    plt.xlabel(feature)


    # Set the y-axis label.
    plt.ylabel("house_price")


    # Display the legend.
    plt.legend()


    # Enable the grid.
    plt.grid(True)


    # Save the plot as a PNG file.
    plt.savefig(
        f"plot_{feature}.png",
        dpi=200,
        bbox_inches="tight"
    )


    # Display the plot.
    plt.show()

    # plt.close()  # close the plot


# Question 1
# Do property prices fall as crime rates rise?
#
# Yes, property prices tend to fall as crime rates rise.
# There is a negative correlation between crime rate and house price.
# However, the relationship is not perfectly linear;
# a polynomial model describes the data somewhat better.


# Question 2
# How much does the property price increase the closer the property is to the center?
#
# In the dataset, distance_to_center means:
# higher values = farther from the center.
#
# The linear regression shows a positive slope:
# higher distance_to_center values are on average associated
# with higher house_price values.
#
# Therefore, prices do not increase with greater proximity to the center,
# but rather with greater distance from the center.
# However, the relationship is relatively weak.


# Question 3
# Do better educational opportunities also affect house prices?
#
# The variable students_per_teacher measures
# the number of students per teacher.
#
# Lower values indicate better educational conditions.
#
# The correlation with house_price is negative:
# more students per teacher are associated with lower house prices.
#
# Therefore, better educational opportunities seem to be
# associated with higher house prices.

