from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Paris"
STUDENT_FIRST_NAME = "PH"


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
        """
        # Initialize and train Random Forest model
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        self.rf_model.fit(X_train, y_train)
        return self.rf_model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ) -> GradientBoostingClassifier:
        """
        Train a Gradient Boosting classifier and store it in self.gb_model.
        """
        # Initialize and train Gradient Boosting model
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        return self.gb_model

    def calculate_feature_importance(
        self, feature_names: List[str], model_type: str = "rf"
    ) -> Dict[str, float]:
        """
        Calculate and store feature importance for the specified model.
        """
        # Select appropriate model
        model = self.rf_model if model_type == "rf" else self.gb_model

        if model is None:
            raise ValueError(f"Model {model_type} has not been trained yet")

        # Get feature importance scores
        importance_scores = model.feature_importances_

        # Create and store dictionary of feature importances
        self.feature_importance[model_type] = dict(
            zip(feature_names, importance_scores)
        )

        return self.feature_importance[model_type]

    def make_prediction(self, X: np.ndarray, model_type: str = "rf") -> np.ndarray:
        """
        Make predictions using the specified model.
        """
        # Select appropriate model
        model = self.rf_model if model_type == "rf" else self.gb_model

        if model is None:
            raise ValueError(f"Model {model_type} has not been trained yet")

        # Make predictions
        return model.predict(X)

    def evaluate_models(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both models using various metrics.
        """
        results = {}

        # Evaluate each model
        for model_type in ["rf", "gb"]:
            y_pred = self.make_prediction(X_test, model_type)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }

            results[model_type] = metrics

        return results


def perform_experiment(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Perform experiment comparing Random Forest and Gradient Boosting.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize ensemble methods
    ensemble = EnsembleMethods()

    # Train models
    ensemble.train_random_forest(X_train, y_train)
    ensemble.train_gradient_boosting(X_train, y_train)

    # Calculate feature importance for both models
    rf_importance = ensemble.calculate_feature_importance(feature_names, "rf")
    gb_importance = ensemble.calculate_feature_importance(feature_names, "gb")

    # Evaluate models
    evaluation_results = ensemble.evaluate_models(X_test, y_test)

    # Prepare feature importance results
    feature_importance_results = {"rf": rf_importance, "gb": gb_importance}

    return evaluation_results, feature_importance_results


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
