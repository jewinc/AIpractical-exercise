from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Cheng"  # TODO: Fill in your last name
STUDENT_FIRST_NAME = "Jewin"  # TODO: Fill in your first name


class EnsembleMethods:
    """
    A class for implementing and comparing different ensemble methods.
    """

    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.feature_importance = {}

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = None,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest classifier and store it in self.rf_model.

        Args:
            X_train: Training features as a NumPy array.
            y_train: Training labels as a NumPy array.
            n_estimators: Number of trees in the forest (default is 100).
            max_depth: Maximum depth of each tree (default is None, meaning nodes are expanded until all leaves are pure or contain less than min_samples_split samples).

        Returns:
            Trained Random Forest model.

        Steps:
            1. Initialize a RandomForestClassifier with the specified `n_estimators` and `max_depth` values.
            2. Train the model using the provided training data (`X_train`, `y_train`).
            3. Store the trained model in `self.rf_model`.
            4. Return the trained model.
        """
        rdf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rdf.fit(X_train, y_train)
        self.rf_model = rdf
        return rdf

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ) -> GradientBoostingClassifier:
        """
        Train a Gradient Boosting classifier and store it in self.gb_model.

        Args:
            X_train: Training features as a NumPy array.
            y_train: Training labels as a NumPy array.
            n_estimators: Number of boosting stages to be run (default is 100).
            learning_rate: Shrinks the contribution of each tree by this factor (default is 0.1).

        Returns:
            Trained Gradient Boosting model.

        Steps:
            1. Initialize a GradientBoostingClassifier with the given `n_estimators` and `learning_rate`.
            2. Train the model using the provided training data (`X_train`, `y_train`).
            3. Store the trained model in `self.gb_model`.
            4. Return the trained model.
        """
        xgboost = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        xgboost.fit(X_train, y_train)
        self.gb_model = xgboost
        return xgboost


    def calculate_feature_importance(
        self, feature_names: List[str], model_type: str = "rf"
    ) -> Dict[str, float]:
        """
        Calculate and store feature importance for the specified model.

        Args:
            feature_names: List of feature names as strings.
            model_type: Type of model to use for calculating feature importance 
                        ('rf' for Random Forest, 'gb' for Gradient Boosting). 
                        Default is 'rf'.

        Returns:
            Dictionary mapping feature names to their importance scores.

        Steps:
            1. Select the appropriate model based on the `model_type` parameter.
               - Use `self.rf_model` for Random Forest.
               - Use `self.gb_model` for Gradient Boosting.
            2. Raise a `ValueError` if the selected model has not been trained yet.
            3. Retrieve the `feature_importances_` attribute from the selected model.
            4. Pair each feature name with its corresponding importance score to 
               create a dictionary.
            5. Store the resulting dictionary in `self.feature_importance[model_type]`.
            6. Return the feature importance dictionary.
        """
        if model_type == 'rf':
            model = self.rf_model
        elif model_type == 'gb':
            model = self.gb_model
        else:
            raise ValueError('unknown model')
        features = model.feature_importances_
        result = {feature_names[i]:features[i] for i in range(len(features))}
        self.feature_importance[model_type] = result
        return result 

    def make_prediction(self, X: np.ndarray, model_type: str = "rf") -> np.ndarray:
        """
        Make predictions using the specified model.

        Args:
            X: Features to make predictions on, as a NumPy array.
            model_type: Type of model to use for predictions ('rf' for Random Forest, 
                        'gb' for Gradient Boosting). Default is 'rf'.

        Returns:
            Array of predictions corresponding to the input features.

        Steps:
            1. Select the appropriate model based on the `model_type` parameter.
            - Use `self.rf_model` for Random Forest.
            - Use `self.gb_model` for Gradient Boosting.
            2. Raise a `ValueError` if the selected model has not been trained yet.
            3. Use the `predict` method of the selected model to generate predictions for `X`.
            4. Return the predictions as a NumPy array.
        """
        if model_type == 'rf':
            model = self.rf_model
        elif model_type == 'gb':
            model = self.gb_model
        else:
            raise ValueError('unknown model')
        
        return model.predict(X)
        

    def evaluate_models(
    self, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both models using various metrics.

        Args:
            X_test: Test features as a NumPy array.
            y_test: Test labels as a NumPy array.

        Returns:
            Dictionary containing evaluation metrics for both models:
            {
                "rf": {
                    "accuracy": <value>,
                    "precision": <value>,
                    "recall": <value>,
                    "f1": <value>
                },
                "gb": {
                    "accuracy": <value>,
                    "precision": <value>,
                    "recall": <value>,
                    "f1": <value>
                }
            }

        Steps:
            1. Initialize an empty dictionary, `results`, to store evaluation metrics for both models.
            2. For each model type ('rf' for Random Forest and 'gb' for Gradient Boosting):
                - Use the `make_prediction` method to generate predictions for `X_test`.
                - Compute the following metrics using `y_test` and the predictions:
                    - Accuracy: Use `accuracy_score`.
                    - Precision: Use `precision_score` with `average="weighted"`.
                    - Recall: Use `recall_score` with `average="weighted"`.
                    - F1 Score: Use `f1_score` with `average="weighted"`.
                - Store the metrics in the `results` dictionary under the key corresponding 
                to the model type ('rf' or 'gb').
            3. Return the `results` dictionary.
        """
        results = {'rf': None, 'gb': None}
        for model_name in ['rf', 'gb']:
            y = self.make_prediction(X=X_test, model_type=model_name)
            results[model_name] = {
                'accuracy': accuracy_score(y_true=y_test, y_pred=y),
                'precision': precision_score(y_true=y_test, y_pred=y, average='weighted'),
                'recall': recall_score(y_true=y_test, y_pred=y, average='weighted'),
                'f1': f1_score(y_true=y_test, y_pred=y, average='weighted')
            }
        return results



def perform_experiment(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Perform experiment comparing Random Forest and Gradient Boosting.

    Args:
        X: Features as a NumPy array.
        y: Labels as a NumPy array.
        feature_names: List of feature names as strings.

    Returns:
        Tuple containing:
        1. Evaluation results for both models:
        {
            "rf": {
                "accuracy": <value>,
                "precision": <value>,
                "recall": <value>,
                "f1": <value>
            },
            "gb": {
                "accuracy": <value>,
                "precision": <value>,
                "recall": <value>,
                "f1": <value>
            }
        }
        2. Feature importance results for both models:
        {
            "rf": {<feature_name>: <importance_score>, ...},
            "gb": {<feature_name>: <importance_score>, ...}
        }

    Steps:
        1. Split the dataset into training and test sets using `train_test_split` 
        with 80% training and 20% testing data.
        2. Create an instance of the `EnsembleMethods` class.
        3. Train both models using:
        - `train_random_forest` for the Random Forest model.
        - `train_gradient_boosting` for the Gradient Boosting model.
        4. Compute feature importance for each model using `calculate_feature_importance`.
        - Store the results in `rf_importance` for Random Forest.
        - Store the results in `gb_importance` for Gradient Boosting.
        5. Evaluate both models using `evaluate_models`.
        - Store the evaluation metrics in `evaluation_results`.
        6. Prepare the feature importance results dictionary:
        - Keys: `"rf"` and `"gb"`.
        - Values: Corresponding importance scores for each feature.
        7. Return a tuple containing:
        - `evaluation_results` (model performance metrics).
        - `feature_importance_results` (feature importance scores).
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8, stratify=y)
    model = EnsembleMethods()
    model.train_random_forest(x_train, y_train)
    model.train_gradient_boosting(x_train, y_train)
    rf_importance = model.calculate_feature_importance(feature_names=feature_names, model_type = 'rf')
    gb_importance = model.calculate_feature_importance(feature_names=feature_names, model_type = 'gb')
    evaluation_results = model.evaluate_models(x_test, y_test)
    feature_importance_results = {
        'rf': rf_importance,
        'gb': gb_importance
    }
    return (evaluation_results, feature_importance_results)

# Example usage:
if __name__ == "__main__":
    # Load sample dataset
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Perform experiment
    evaluation_results, feature_importance_results = perform_experiment(
            X, y, feature_names
        )

    print("\nEvaluation Results:")
    for model, metrics in evaluation_results.items():
        print(f"\n{model} Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    print("\nTop 5 Important Features:")
    for model, importance in feature_importance_results.items():
        print(f"\n{model} Important Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        for feature, importance_score in sorted_features:
            print(f"{feature}: {importance_score:.4f}")
