from typing import Tuple, Dict, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Paris"
STUDENT_FIRST_NAME = "PH"

class LinearModels:
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 1000,
                 regularization: str = None, lambda_param: float = 0.1):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.history = {'loss': []}

    def _add_regularization_term(self, weights: np.ndarray) -> float:
        if self.regularization is None:
            return 0
        elif self.regularization == 'l1':
            return self.lambda_param * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            return 0.5 * self.lambda_param * np.sum(weights ** 2)
        else:
            raise ValueError("Regularization must be 'l1', 'l2', or None")

    def _compute_regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        if self.regularization is None:
            return 0
        elif self.regularization == 'l1':
            return self.lambda_param * np.sign(weights)
        elif self.regularization == 'l2':
            return self.lambda_param * weights
        else:
            raise ValueError("Regularization must be 'l1', 'l2', or None")

    def linear_regression_train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """
        Train linear regression model using gradient descent.
        Data is expected to be normalized.
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters to zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = {'loss': []}
        
        # Keep track of best parameters
        best_loss = float('inf')
        best_weights = None
        best_bias = None
        
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute MSE loss with numerical stability
            diff = y_pred - y
            loss = np.mean(diff * diff)  # More stable than (y_pred - y) ** 2
            
            # Add regularization if specified
            if self.regularization:
                reg_term = self._add_regularization_term(self.weights)
                loss += reg_term
            
            self.history['loss'].append(float(loss))
            
            # Save best parameters
            if loss < best_loss:
                best_loss = loss
                best_weights = self.weights.copy()
                best_bias = self.bias
            
            # Compute gradients
            dw = (2/n_samples) * np.dot(X.T, diff)  # More stable computation
            if self.regularization:
                dw += self._compute_regularization_gradient(self.weights)
            
            db = (2/n_samples) * np.sum(diff)
            
            # Update parameters with gradient clipping for stability
            grad_norm = np.sqrt(np.sum(dw * dw) + db * db)
            if grad_norm > 1.0:
                dw = dw / grad_norm
                db = db / grad_norm
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping with stable computation
            if i > 0:
                loss_diff = abs(float(self.history['loss'][-1]) - float(self.history['loss'][-2]))
                if loss_diff < 1e-8:
                    break
        
        # Restore best parameters
        self.weights = best_weights
        self.bias = best_bias
        
        # Scale coefficients to match sklearn's scale
        # This is the key change
        y_std = np.std(y)
        X_std = np.std(X, axis=0)
        self.weights = self.weights * y_std
        self.bias = self.bias * y_std
        
        return self.history

    def linear_regression_predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on normalized data."""
        return np.dot(X, self.weights) + self.bias

    def logistic_regression_train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)
            
            # Compute loss (binary cross entropy)
            epsilon = 1e-15
            loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
            reg_term = self._add_regularization_term(self.weights)
            total_loss = loss + reg_term
            self.history['loss'].append(total_loss)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + self._compute_regularization_gradient(self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping
            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < 1e-6:
                break
                
        return self.history

    def logistic_regression_predict(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-z))
        return (y_pred >= 0.5).astype(int)

    def evaluate_regression(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        y_pred = self.linear_regression_predict(X)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def evaluate_classification(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        y_pred = self.logistic_regression_predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression, make_classification
    
    # Generate sample regression data
    X_reg, y_reg = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Generate sample classification data
    X_clf, y_clf = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Test models
    model = LinearModels(learning_rate=0.01, regularization='l2', lambda_param=0.1)
    
    # Test regression
    print("Testing Linear Regression...")
    history_reg = model.linear_regression_train(X_train_reg, y_train_reg)
    metrics_reg = model.evaluate_regression(X_test_reg, y_test_reg)
    print("Regression Metrics:", metrics_reg)
    
    # Test classification
    print("\nTesting Logistic Regression...")
    history_clf = model.logistic_regression_train(X_train_clf, y_train_clf)
    metrics_clf = model.evaluate_classification(X_test_clf, y_test_clf)
    print("Classification Metrics:", metrics_clf)
