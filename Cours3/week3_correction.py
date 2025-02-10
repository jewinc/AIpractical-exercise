from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

STUDENT_LAST_NAME = "Paris"
STUDENT_FIRST_NAME = "PH"


class DecisionTreeAnalyzer:
    def __init__(self):
        self.tree = None
        self.feature_names = None

    def calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy for a target array."""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for a target array."""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities**2)
        return gini

    def calculate_information_gain(
        self,
        y_parent: np.ndarray,
        y_children: List[np.ndarray],
        criterion: str = "entropy",
    ) -> float:
        """Calculate information gain for a split."""
        if criterion not in ["entropy", "gini"]:
            raise ValueError("Criterion must be either 'entropy' or 'gini'")

        # Handle empty arrays
        if len(y_parent) == 0 or not y_children:
            return 0.0

        # Calculate parent impurity
        if criterion == "entropy":
            parent_impurity = self.calculate_entropy(y_parent)
        else:
            parent_impurity = self.calculate_gini(y_parent)

        # Calculate weighted average of children impurities
        n_parent = len(y_parent)
        weighted_child_impurity = 0.0

        total_children = sum(len(child) for child in y_children)
        if total_children == 0:
            return 0.0

        for child in y_children:
            if len(child) > 0:  # Only consider non-empty children
                weight = len(child) / total_children
                if criterion == "entropy":
                    impurity = self.calculate_entropy(child)
                else:
                    impurity = self.calculate_gini(child)
                weighted_child_impurity += weight * impurity

        # Normalize information gain
        information_gain = parent_impurity - weighted_child_impurity
        return max(0.0, min(1.0, information_gain))  # Ensure value is between 0 and 1

    def analyze_split_quality(
        self, X: pd.DataFrame, y: np.ndarray, feature: str, threshold: float
    ) -> Dict[str, float]:
        """Analyze the quality of a potential split."""
        if len(y) == 0 or feature not in X.columns:
            return {
                "entropy": 0.0,
                "gini": 0.0,
                "information_gain_entropy": 0.0,
                "information_gain_gini": 0.0,
            }

        # Create masks for split
        left_mask = X[feature] <= threshold
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        # Calculate base metrics
        entropy = self.calculate_entropy(y)
        gini = self.calculate_gini(y)

        # Calculate information gains
        entropy_gain = self.calculate_information_gain(y, [y_left, y_right], "entropy")
        gini_gain = self.calculate_information_gain(y, [y_left, y_right], "gini")

        # Handle edge cases and validate results
        metrics = {
            "entropy": max(0.0, min(1.0, entropy)),
            "gini": max(0.0, min(1.0, gini)),
            "information_gain_entropy": max(0.0, min(1.0, entropy_gain)),
            "information_gain_gini": max(0.0, min(1.0, gini_gain)),
        }

        return metrics

    def fit_and_evaluate(
        self, X: pd.DataFrame, y: np.ndarray, params: Dict[str, Union[str, int, float]]
    ) -> Dict[str, float]:
        """Fit and evaluate a decision tree with given parameters."""
        try:
            # Validate input parameters
            required_params = ["criterion", "max_depth", "min_samples_split"]
            if not all(param in params for param in required_params):
                raise ValueError("Missing required parameters")

            # Split data with stratification
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create and fit the tree
            self.tree = DecisionTreeClassifier(**params, random_state=42)
            self.tree.fit(X_train, y_train)
            self.feature_names = X.columns.tolist()

            # Calculate metrics
            train_score = self.tree.score(X_train, y_train)
            val_score = self.tree.score(X_val, y_val)

            metrics = {
                "train_accuracy": max(0.0, min(1.0, train_score)),
                "val_accuracy": max(0.0, min(1.0, val_score)),
                "tree_depth": self.tree.get_depth(),
                "n_leaves": self.tree.get_n_leaves(),
                "n_features": len(self.feature_names),
                "n_samples_train": len(X_train),
                "n_samples_val": len(X_val),
            }

            return metrics

        except Exception as e:
            raise ValueError(f"Error in model evaluation: {str(e)}")

    def visualize_tree(self, feature_names: List[str], class_names: List[str]) -> None:
        """Visualize the trained decision tree."""
        if self.tree is None:
            raise ValueError("Tree must be fitted before visualization")

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.title("Decision Tree Visualization")
        plt.tight_layout()
        plt.show()

    def analyze_feature_importance(self) -> pd.DataFrame:
        """Analyze and return feature importance scores."""
        if self.tree is None or self.feature_names is None:
            raise ValueError("Tree must be fitted before analyzing feature importance")

        # Get raw importance scores
        importance_scores = self.tree.feature_importances_

        # Normalize scores to sum to 1
        normalized_scores = importance_scores / np.sum(importance_scores)

        # Create DataFrame with both raw and normalized scores
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_raw": importance_scores,
                "importance_normalized": normalized_scores,
            }
        )

        # Sort by normalized importance in descending order
        importance_df = importance_df.sort_values(
            "importance_normalized", ascending=False
        )

        # Add rank column
        importance_df["rank"] = range(1, len(importance_df) + 1)

        # Add cumulative importance
        importance_df["cumulative_importance"] = importance_df[
            "importance_normalized"
        ].cumsum()

        return importance_df

    def compare_pruning_methods(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        methods: List[Dict[str, Union[str, int, float]]],
    ) -> pd.DataFrame:
        """Compare different pruning methods."""
        results = []

        for params in methods:
            metrics = self.fit_and_evaluate(X, y, params)
            results.append(
                {
                    "criterion": params["criterion"],
                    "max_depth": params["max_depth"],
                    "min_samples_split": params["min_samples_split"],
                    "train_accuracy": metrics["train_accuracy"],
                    "val_accuracy": metrics["val_accuracy"],
                    "tree_depth": metrics["tree_depth"],
                    "n_leaves": metrics["n_leaves"],
                    "n_features": metrics["n_features"],
                    "n_samples_train": metrics["n_samples_train"],
                    "n_samples_val": metrics["n_samples_val"],
                }
            )

        return pd.DataFrame(results)
