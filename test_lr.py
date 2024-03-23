import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from linear_regression import LinearRegression  # Assuming your LinearRegression class is defined in linear_regression.py

# Load California Housing Dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Number of iterations
num_iterations = 20

# Arrays to store RMSE values
train_rmse_custom = np.zeros(num_iterations)
test_rmse_custom = np.zeros(num_iterations)

train_rmse_sklearn = np.zeros(num_iterations)
test_rmse_sklearn = np.zeros(num_iterations)

for i in range(num_iterations):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Create and train the Custom Linear Regression model
    custom_model = LinearRegression(n_features=X_train.shape[1])
    custom_model.fit(X_train, y_train)

    # Evaluate the Custom Linear Regression model
    train_results_custom = custom_model.evaluate(X_train, y_train)
    train_rmse_custom[i] = np.sqrt(train_results_custom[1])

    test_results_custom = custom_model.evaluate(X_test, y_test)
    test_rmse_custom[i] = np.sqrt(test_results_custom[1])

    # Create and train the scikit-learn Linear Regression model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)

    # Evaluate the scikit-learn Linear Regression model
    train_predictions_sklearn = sklearn_model.predict(X_train)
    train_rmse_sklearn[i] = np.sqrt(np.mean((train_predictions_sklearn - y_train) ** 2))

    test_predictions_sklearn = sklearn_model.predict(X_test)
    test_rmse_sklearn[i] = np.sqrt(np.mean((test_predictions_sklearn - y_test) ** 2))

# Calculate and print the mean and standard deviation of RMSE for Custom Linear Regression
mean_train_rmse_custom = np.mean(train_rmse_custom)
std_train_rmse_custom = np.std(train_rmse_custom)

mean_test_rmse_custom = np.mean(test_rmse_custom)
std_test_rmse_custom = np.std(test_rmse_custom)

print("Custom Linear Regression Results:")
print(f"Mean Training RMSE: {mean_train_rmse_custom}")
print(f"Standard Deviation of Training RMSE: {std_train_rmse_custom}")
print(f"Mean Testing RMSE: {mean_test_rmse_custom}")
print(f"Standard Deviation of Testing RMSE: {std_test_rmse_custom}")

# Calculate and print the mean and standard deviation of RMSE for scikit-learn Linear Regression
mean_train_rmse_sklearn = np.mean(train_rmse_sklearn)
std_train_rmse_sklearn = np.std(train_rmse_sklearn)

mean_test_rmse_sklearn = np.mean(test_rmse_sklearn)
std_test_rmse_sklearn = np.std(test_rmse_sklearn)

print("\nScikit-learn Linear Regression Results:")
print(f"Mean Training RMSE: {mean_train_rmse_sklearn}")
print(f"Standard Deviation of Training RMSE: {std_train_rmse_sklearn}")
print(f"Mean Testing RMSE: {mean_test_rmse_sklearn}")
print(f"Standard Deviation of Testing RMSE: {std_test_rmse_sklearn}")
