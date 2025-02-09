from typing import Dict, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Cheng"  
STUDENT_FIRST_NAME = "Jewin"  


class LinearModels:
    """
    Implementation of linear regression and logistic regression from scratch.
    Includes regularization options and various evaluation metrics.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        regularization: str = None,
        lambda_param: float = 0.1,
    ):
        """
        Initialize the model with hyperparameters.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations for gradient descent
            regularization: Type of regularization ('l1', 'l2', or None)
            lambda_param: Regularization strength
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.history = {"loss": [], 'loss_params': []}

    def _add_regularization_term(self, weights: np.ndarray) -> float:
        """
        Calculate regularization term based on specified regularization type.

        Args:
            weights: Model weights

        Returns:
            Regularization term
        """
        # TODO: Implement the regularization term for L1 and L2 regularization.
        # Tips:
        # 1. If no regularization type is specified (i.e., `self.regularization is None`), return 0.
        # 2. For L1 regularization ('l1'), the term is proportional to the sum of the absolute values of the weights.
        #    Use: np.abs(weights) and np.sum() to compute this.
        # 3. For L2 regularization ('l2'), the term is proportional to the sum of the squared weights.
        #    Remember to include a scaling factor of 0.5 for consistency with the standard definition.
        # 4. Use `self.lambda_param` to scale the regularization term for both L1 and L2 cases.
        # 5. Raise an error for unsupported regularization types to ensure robustness.

        # Pseudo-code:
        # if no regularization:
        #     return 0
        # if L1 regularization:
        #     return lambda_param * sum of absolute weights
        # if L2 regularization:
        #     return 0.5 * lambda_param * sum of squared weights
        # else:
        #     raise ValueError for unsupported regularization types

        if self.regularization == None:
            return 0
        elif self.regularization == 'l1':
            return self.lambda_param * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            return 0.5 * self.lambda_param * np.sum(np.square(weights))
        else:
            raise ValueError('unsupported regularization types')


    def _compute_regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute gradient of regularization term.

        Args:
            weights: Model weights

        Returns:
            Gradient of regularization term
        """
        # TODO: Implement the gradient of the regularization term for L1 and L2 regularization.
        # Tips:
        # 1. If no regularization type is specified (i.e., `self.regularization is None`), return 0.
        #    Note: Return type must match the input type (`np.ndarray`), so ensure compatibility.
        # 2. For L1 regularization ('l1'), the gradient is proportional to the sign of each weight.
        #    Use: `np.sign(weights)` to compute this.
        # 3. For L2 regularization ('l2'), the gradient is proportional to the weights themselves.
        #    This is often referred to as a "ridge penalty".
        # 4. Use `self.lambda_param` to scale the gradient for both L1 and L2 cases.
        # 5. Raise an error for unsupported regularization types to ensure robustness.

        # Pseudo-code:
        # if no regularization:
        #     return an array of zeros with the same shape as weights
        # if L1 regularization:
        #     return lambda_param * sign of weights
        # if L2 regularization:
        #     return lambda_param * weights
        # else:
        #     raise ValueError for unsupported regularization types

        # Hint for students:
        # - Use `np.zeros_like(weights)` to create a zero array matching the shape of weights when no regularization is applied.

        if self.regularization == None:
            return np.zeros_like(weights)
        elif self.regularization == 'l1':
            return self.lambda_param * np.sign(weights)
        elif self.regularization == 'l2':
            return self.lambda_param * weights
        else:
            raise ValueError('unsupported regularization types')


    def linear_regression_train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """
        Train linear regression model using gradient descent.

        Args:
            X: Training features
            y: Target values

        Returns:
            Dictionary containing training history
        """
        # TODO: Implement linear regression training using gradient descent.
        # Steps:
        # 1. Initialize weights and bias to zeros.
        #    - Hint: Use `np.zeros(n_features)` for weights and a scalar (0.0) for bias.
        #    - Save the initial values of weights and bias for later reference.
        #
        # 2. Loop through the specified number of iterations (`self.max_iterations`):
        #    - Compute predictions: Use `np.dot(X, self.weights) + self.bias`.
        #    - Calculate the Mean Squared Error (MSE) loss:
        #        * Use `(y_pred - y) ** 2` but ensure numerical stability.
        #        * Add the regularization term (if applicable) using `_add_regularization_term`.
        #        * Append the loss value to `self.history['loss']`.
        #
        # 3. Compute gradients for weights (`dw`) and bias (`db`):
        #    - For weights: Use `(2/n_samples) * np.dot(X.T, diff)`, where `diff = y_pred - y`.
        #    - Add the regularization gradient for weights using `_compute_regularization_gradient`.
        #    - For bias: Use `(2/n_samples) * np.sum(diff)`.
        #
        # 4. Update weights and bias:
        #    - Use gradient descent with `self.learning_rate`.
        #    - Optionally, clip gradients to ensure stability.
        #
        # 5. Implement early stopping:
        #    - Stop the loop if the change in loss between iterations is very small (e.g., < 1e-8).
        #
        # 6. Save the best weights and bias (based on lowest loss encountered).
        #
        # 7. (Optional) Scale the weights and bias to account for normalized data:
        #    - Adjust the weights and bias to match the scale of `y` and `X`.

        # Hint: You will need to carefully combine forward computation, loss calculation, 
        # gradient computation, and parameter updates to ensure convergence.

        # Pseudo-code outline:
        # n_samples, n_features = X.shape
        # Initialize weights and bias
        # Loop over max_iterations:
        #     Compute predictions
        #     Compute loss and append to history
        #     Compute gradients
        #     Update weights and bias
        #     Check for early stopping
        # Save the best weights and bias
        # Scale weights and bias to match data (optional)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        bw = self.weights
        bb = self.bias
        for i in range(self.max_iterations):
            #Compute prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            #Compute loss and append to history
            loss = np.mean((y_pred - y)**2) + self._add_regularization_term(weights= self.weights)
            self.history['loss'].append(loss)
            self.history['loss_params'].append((self.weights, self.bias))
            
            #Compute gradients
            diff = y_pred - y
            dw = (2/n_samples) * np.dot(X.T, diff) + self._compute_regularization_gradient(weights= self.weights)
            db = (2/n_samples) * np.sum(diff)

            #Update weights and bias
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

            #Early stopping
            if loss < 1e-8:
                break
    
        #return history and update weight and bias with best perf
        i = 0
        min_loss = self.history['loss'][0]
        for j in range(len(self.history['loss'])):
            if self.history['loss'][j] < min_loss:
                min_loss = self.history['loss'][j]
                i = j
        self.weights, self.bias = self.history['loss_params'][i]
        return self.history 


    def linear_regression_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained linear regression model.

        Args:
            X: Features to predict

        Returns:
            Predicted values
        """
        # TODO: Implement prediction for linear regression.
        # Tips:
        # 1. Use the learned weights (`self.weights`) and bias (`self.bias`) to compute predictions.
        # 2. Apply the linear regression formula:
        #    - For each sample in X, the prediction is the dot product of the sample and the weights, 
        #      plus the bias: `np.dot(X, self.weights) + self.bias`.
        # 3. Ensure that the weights and bias have been properly trained before calling this method.

        # Hint:
        # - Use `np.dot` for efficient matrix-vector multiplication.
        # - If you encounter incorrect results, double-check that `self.weights` and `self.bias` are initialized correctly.

        if type(self.weights) != None and type(self.bias) != None:
            bias = np.dot(X, self.weights) + self.bias
            return bias
        else:
            raise ValueError('weights and bias not init')

    def logistic_regression_train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """
        Train logistic regression model using gradient descent.

        Args:
            X: Training features
            y: Target values (binary)

        Returns:
            Dictionary containing training history
        """
        # TODO: Implement logistic regression training using gradient descent.
        # Steps:
        # 1. Initialize weights and bias to zeros:
        #    - Use `np.zeros(n_features)` for weights.
        #    - Initialize bias as a scalar (0.0).
        #
        # 2. Define the sigmoid activation function:
        #    - Formula: sigmoid(z) = 1 / (1 + exp(-z)).
        #    - Use `np.exp` for the exponential computation and apply clipping (e.g., `np.clip(z, -250, 250)`) 
        #      to avoid numerical instability for very large or very small values of `z`.
        #
        # 3. Loop through `self.max_iterations`:
        #    - Compute `z = np.dot(X, self.weights) + self.bias` (logits).
        #    - Apply the sigmoid function to compute predictions (`y_pred`).
        #
        # 4. Compute binary cross-entropy loss:
        #    - Formula: -mean(y * log(y_pred) + (1 - y) * log(1 - y_pred)).
        #    - Add a small constant (`epsilon = 1e-15`) to `y_pred` to avoid taking the log of 0.
        #    - Include the regularization term using `_add_regularization_term`.
        #
        # 5. Compute gradients:
        #    - For weights: `(1/n_samples) * np.dot(X.T, (y_pred - y))`.
        #    - Add the regularization gradient using `_compute_regularization_gradient`.
        #    - For bias: `(1/n_samples) * np.sum(y_pred - y)`.
        #
        # 6. Update weights and bias using the gradients and `self.learning_rate`.
        #
        # 7. Early stopping:
        #    - Stop the loop if the difference in loss between consecutive iterations is very small (e.g., < 1e-6).
        #
        # 8. Append total loss (cross-entropy + regularization term) to `self.history['loss']`.

        # Hints:
        # - Carefully combine forward computation, loss calculation, gradient computation, and parameter updates.
        # - Numerical stability is critical, especially when computing the logarithm in the loss function.
        # - Early stopping improves efficiency by preventing unnecessary iterations.

        # Pseudo-code:
        # n_samples, n_features = X.shape
        # Initialize weights and bias
        # Define sigmoid function
        # Loop over max_iterations:
        #     Compute logits (z)
        #     Compute predictions using sigmoid
        #     Compute loss and append to history
        #     Compute gradients for weights and bias
        #     Update weights and bias
        #     Check for early stopping
        # Return training history

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        f_sigm = lambda z: 1/ (1 + np.clip(np.exp(-z), -250, 250))
        for i in range(self.max_iterations):

            #Compute logistic and prediction
            y_pred = f_sigm(np.dot(X, self.weights) + self.bias)
            
            #Compute loss and append to history
            reg_term = self._add_regularization_term(self.weights)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) + 1e-15 + reg_term
            self.history['loss'].append((loss, reg_term))
            self.history['loss_params'].append((self.weights, self.bias))
            
            #Compute gradients
            diff = y_pred - y
            dw = (1/n_samples) * np.dot(X.T, diff) + self._compute_regularization_gradient(weights= self.weights)
            db = (1/n_samples) * np.sum(diff)

            #Update weights and bias
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

            #Early stopping
            if loss < 1e-6:
                break
    
        #return history and update weight and bias with best perf
        i = 0
        min_loss = self.history['loss'][0]
        for j in range(len(self.history['loss'])):
            if self.history['loss'][j] < min_loss:
                min_loss = self.history['loss'][j]
                i = j
        self.weights, self.bias = self.history['loss_params'][i]
        return self.history 


    def logistic_regression_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained logistic regression model.

        Args:
            X: Features to predict

        Returns:
            Predicted classes (0 or 1)
        """
        # TODO: Implement prediction for logistic regression.
        # Tips:
        # 1. Compute logits (`z`) using the formula:
        #    z = np.dot(X, self.weights) + self.bias
        # 2. Apply the sigmoid activation function to transform logits into probabilities:
        #    sigmoid(z) = 1 / (1 + exp(-z)).
        #    - Use `np.exp` for the exponential calculation.
        #    - Ensure numerical stability if needed (e.g., by clipping `z` to avoid overflow).
        # 3. Convert probabilities (`y_pred`) to binary class predictions:
        #    - Use a threshold of 0.5: If `y_pred >= 0.5`, classify as 1; otherwise, classify as 0.
        #    - You can achieve this using `(y_pred >= 0.5).astype(int)`.

        # Hint:
        # - Ensure that the model's weights (`self.weights`) and bias (`self.bias`) have been trained before calling this method.

        # Pseudo-code outline:
        # Compute logits (z) using np.dot
        # Apply sigmoid to compute probabilities
        # Convert probabilities to binary class predictions (0 or 1)
        
        z = np.dot(X, self.weights) + self.bias
        f_sigm = lambda z: 1/ (1 + np.clip(np.exp(-z), -250, 250))
        y_pred = f_sigm(z)
        return (y_pred >= 0.5).astype(int)



    def evaluate_regression(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model performance.

        Args:
            X: Features
            y_true: True target values

        Returns:
            Dictionary containing evaluation metrics:
            - Mean Squared Error (MSE)
            - Root Mean Squared Error (RMSE)
            - Coefficient of Determination (R^2)
        """
        # TODO: Implement regression evaluation metrics.
        # Steps:
        # 1. Use `self.linear_regression_predict` to compute predictions (`y_pred`) for the input `X`.
        # 2. Calculate Mean Squared Error (MSE):
        #    - Formula: mean((y_true - y_pred) ** 2)
        #    - Hint: Use the `mean_squared_error` function from `sklearn.metrics` for this step.
        # 3. Calculate Root Mean Squared Error (RMSE):
        #    - Formula: sqrt(MSE)
        #    - Hint: Use `np.sqrt` to compute the square root.
        # 4. Calculate the Coefficient of Determination (R^2):
        #    - Formula: 1 - sum((y_true - y_pred) ** 2) / sum((y_true - mean(y_true)) ** 2)
        #    - This measures how well the predictions explain the variance in the target values.

        # Hint:
        # - Make sure `self.linear_regression_predict` has been implemented correctly, as it is a dependency for this method.

        # Pseudo-code:
        # 1. Compute predictions using `linear_regression_predict`.
        # 2. Calculate MSE.
        # 3. Calculate RMSE.
        # 4. Calculate R^2.
        # 5. Return a dictionary with the computed metrics.
        # return {
        #     'mse': mse,
        #     'rmse': rmse,
        #     'r2': r2
        # }
        y_pred = self.linear_regression_predict(X)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return { 'mse': mse,
                'rmse': rmse,
                'r2': r2
                }


    def evaluate_classification(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification model performance.

        Args:
            X: Features
            y_true: True class labels

        Returns:
            Dictionary containing evaluation metrics:
            - Accuracy
            - Precision
            - Recall
            - F1 Score
        """        
        # TODO: Implement classification evaluation metrics.
        # Steps:
        # 1. Use `self.logistic_regression_predict` to compute predictions (`y_pred`) for the input `X`.
        #    - Ensure that `logistic_regression_predict` is implemented and returns binary predictions (0 or 1).
        # 2. Calculate the following metrics using `sklearn.metrics`:
        #    - Accuracy: Fraction of correct predictions over total predictions.
        #        * Use `accuracy_score(y_true, y_pred)`.
        #    - Precision: Proportion of positive predictions that are actually correct.
        #        * Use `precision_score(y_true, y_pred)`.
        #    - Recall: Proportion of actual positives that are correctly predicted.
        #        * Use `recall_score(y_true, y_pred)`.
        #    - F1 Score: Harmonic mean of precision and recall.
        #        * Use `f1_score(y_true, y_pred)`.

        # Hint:
        # - Ensure `y_pred` contains binary values (0 or 1). If predictions are unexpected, debug `logistic_regression_predict`.
        # - Use the metrics from `sklearn.metrics` to simplify computation.

        # Pseudo-code:
        # 1. Compute predictions using `logistic_regression_predict`.
        # 2. Calculate each metric (accuracy, precision, recall, F1).
        # 3. Return a dictionary with the computed metrics.
        # result = {
        #     'accuracy': 0.0,
        #     'precision': 0.0,
        #     'recall': 0.0,
        #     'f1': 0.0
        # }
        result = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        y_pred = self.logistic_regression_predict(X)
        result['accuracy'] = accuracy_score(y_true, y_pred)
        result['precision'] = precision_score(y_true, y_pred)
        result['recall'] = recall_score(y_true, y_pred)
        result['f1'] = f1_score(y_true, y_pred)
        return result



if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression

    # Generate sample regression data
    X_reg, y_reg = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Generate sample classification data
    X_clf, y_clf = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # Initialize model
    model = LinearModels(learning_rate=0.01, regularization="l2", lambda_param=0.1)
    

    # Example outputs (students will uncomment these after completing the TODOs):
    print("Testing Linear Regression...")
    model.linear_regression_train(X_train_reg, y_train_reg)
    print("Regression Metrics:", model.evaluate_regression(X_test_reg, y_test_reg))
    
    # Init the model again, removing linear_regression result
    model = LinearModels(learning_rate=0.01, regularization="l2", lambda_param=0.1)
    print("\nTesting Logistic Regression...")
    model.logistic_regression_train(X_train_clf, y_train_clf)
    print("Classification Metrics:", model.evaluate_classification(X_test_clf, y_test_clf))
