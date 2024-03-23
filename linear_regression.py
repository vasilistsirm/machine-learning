import numpy as np


class LinearRegression:
    def __init__(self, n_features):
        self.w = None  # Βάρη
        self.b = None  # Όρος μετατόπισης
        self.n_features = n_features  # Πλήθος χαρακτηριστικών
        self.N = 0  # Πλήθος δεδομένων

    def initialize_weights(self):
        self.w = np.zeros(self.n_features)
        self.b = 0.0

    def fit(self, X, y):
        # Επιβεβαίωση ότι X και y είναι numpy arrays και οι διαστάσεις είναι συμβατές
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")

        self.N = X.shape[0]
        self.initialize_weights()

        # Δημιουργία επιπλέον στήλης για τον όρο μετατόπισης
        X_with_bias = np.c_[X, np.ones(self.N)]

        # Υπολογισμός παραμέτρων θ χρησιμοποιώντας τις κανονικές εξισώσεις
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

        # Αποθήκευση στις ιδιότητες w και b της κλάσης
        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X):
        if self.w is None or self.b is None:
            raise Exception("Model not trained. Fit the model before making predictions.")

        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")

        # Add bias column to X
        X_with_bias = np.c_[X, np.ones(X.shape[0])]

        predictions = X_with_bias @ np.hstack((self.w, self.b))
        return predictions

    def evaluate(self, X, y):
        if self.w is None or self.b is None:
            raise Exception("Model not trained. Fit the model before evaluation.")

        # Make predictions
        predictions = self.predict(X)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((predictions - y) ** 2)

        return predictions, mse
