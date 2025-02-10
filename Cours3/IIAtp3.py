from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

#! DO NOT ADD OR MODIFY IMPORTS !

STUDENT_LAST_NAME = "Cheng"  
STUDENT_FIRST_NAME = "Jewin"


class DecisionTreeAnalyzer:
    """
    A class for analyzing and visualizing decision trees with different parameters
    and evaluation metrics.
    """

    def __init__(self):
        self.tree = None
        self.feature_names = None

    def calculate_entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy for a target array.

        Args:
            y: Target array of class labels.

        Returns:
            Entropy value.

        Hint: Use the formula for entropy from information theory.
            1. Find the unique class labels and their counts.
            2. Compute the probability of each class.
            3. Combine probabilities using the entropy formula.
        """
        # TODO: Implement entropy calculation
        # Step 0: Edge case -> No data points, entropy is zero.
        # Step 1: Identify unique labels and their counts.
        # Step 2: Calculate the probability of each label.
        #         Hint: Print intermediate results like unique labels, counts, and probabilities for debugging.
        # Step 3: Use probabilities to calculate entropy.
        #         (Think about how logarithms behave with probabilities.)
        # Step 4: Replace 'pass' with the correct implementation.
        entropy = 0
        if len(y) > 0:
            classes, count = np.unique(y, return_counts=True)
            for i in range(len(classes)):
                entropy += count[i]/len(y) * np.log2(count[i]/len(y))
            entropy = -entropy
        return entropy


    def calculate_gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for a target array.

        Args:
            y: Target array of class labels.

        Returns:
            Gini impurity value.

        Hint: Use the formula for Gini impurity from information theory.
            1. Find the unique class labels and their counts.
            2. Compute the probability of each class.
            3. Combine probabilities using the Gini impurity formula.
        """
        # Implement Gini impurity calculation
        # Step 0: Edge case -> No data points, Gini impurity is zero.
        # Step 1: Identify unique labels and their counts.
        # Step 2: Calculate the probability of each label.
        #         Hint: Print intermediate results like unique labels, counts, and probabilities for debugging.
        # Step 3: Use probabilities to calculate Gini impurity.
        #         (Think about how probabilities contribute to impurity.)
        # Step 4: Replace 'pass' with the correct implementation.
        gini = 0
        if len(y) > 0:
            classes, count = np.unique(y, return_counts=True)
            for i in range(len(classes)):
                gini += (count[i]/len(y))**2
            gini = 1 - gini
        return gini


    def calculate_information_gain(
        self,
        y_parent: np.ndarray,
        y_children: List[np.ndarray],
        criterion: str = "entropy",
    ) -> float:
        """
        Calculate information gain for a split.

        Args:
            y_parent: Parent node target array.
            y_children: List of children node target arrays.
            criterion: Splitting criterion ('entropy' or 'gini').

        Returns:
            Information gain value.

        Hint: Use the information gain formula from decision tree theory.
            1. Calculate the impurity of the parent node.
            2. Compute a weighted average of the impurities of the child nodes.
            3. Subtract the weighted child impurity from the parent impurity.
            4. Normalize the result between 0 and 1, if necessary.
        """
        # TODO: Implement information gain calculation
        # Step 0: Check for valid criterion ('entropy' or 'gini').
        #         Raise a ValueError if an invalid criterion is provided.
        # Step 1: Handle edge cases:
        #         - If the parent array is empty or no children are provided, return 0.0.
        # Step 2: Calculate the impurity of the parent node using the selected criterion.
        #         Hint: Use self.calculate_entropy or self.calculate_gini.
        # Step 3: Compute a weighted average of the children impurities.
        #         - Iterate over non-empty child arrays and compute their contributions.
        # Step 4: Subtract the weighted average of child impurities from the parent impurity.
        #         Hint: Think about how the weighted average impacts the split quality.
        # Step 5: Return the information gain, ensuring it is normalized between 0 and 1.
        if len(y_parent)<= 0 or len(y_children) <= 0:
            return 0.0
        else:
            if criterion == 'gini':
                f = self.calculate_gini
            elif criterion == 'entropy':
                f = self.calculate_entropy
            else:
                raise ValueError('not valid criterion')
            
            g_parent = f(y_parent)
            w = 0.0
            for children in y_children:
                if len(children) >0:
                    local_w = len(children)/len(y_children)
                    w+= local_w * f(children)
            return g_parent - w
                

    def analyze_split_quality(
        self, X: pd.DataFrame, y: np.ndarray, feature: str, threshold: float
    ) -> Dict[str, float]:
        """
        Analyze the quality of a potential split for a numerical feature.

        Args:
            X: Feature DataFrame.
            y: Target array.
            feature: Feature name to analyze.
            threshold: Splitting threshold value.

        Returns:
            Dictionary containing entropy, Gini impurity, and information gain metrics.

        Hint: Evaluate the quality of a split by calculating:
            1. The entropy and Gini impurity of the parent node.
            2. The information gain for splitting the parent into two child nodes.
        """
        # TODO: Implement split quality analysis
        metrics = {
            "entropy": 0.0,
            "gini": 0.0,
            "information_gain_entropy": 0.0,
            "information_gain_gini": 0.0,
        }
        # Step 0: Handle edge cases:
        #         - If the target array is empty or the feature is invalid, return default values (e.g., 0.0 for all metrics).
        # Step 1: Create masks for splitting data into left and right groups based on the threshold.
        #         Hint: Compare feature values to the threshold (e.g., X[feature] <= threshold).
        # Step 2: Use the masks to split `y` into left and right groups.
        # Step 3: Calculate base metrics:
        #         - Compute the entropy and Gini impurity for the parent node.
        #         Hint: Use self.calculate_entropy and self.calculate_gini.
        # Step 4: Calculate information gains:
        #         - Use self.calculate_information_gain for both 'entropy' and 'gini' criteria.
        # Step 5: Normalize the results (if necessary) to ensure values remain valid (e.g., between 0 and 1).
        # Step 6: Replace 'pass' with the correct implementation.
        rows, columns = X.shape
        if not(rows <= 0 or columns <= 0 or not(feature in X.columns)):
            l_mask, r_mask = X[feature] <= threshold, X[feature] > threshold
            l_y, r_y = y[l_mask], y[r_mask]
            if len(l_y) <= 0 or len(r_y) <=0:
                return metrics
            
            ig_gini = self.calculate_information_gain(y, [l_y, r_y], criterion= 'gini')
            ig_entropy = self.calculate_information_gain(y, [l_y, r_y])
            metrics['entropy'] = self.calculate_entropy(y)
            metrics['gini'] = self.calculate_gini(y)
            metrics['information_gain_entropy'] = ig_entropy
            metrics['information_gain_gini'] = ig_gini
        return metrics

    def fit_and_evaluate(
        self, X: pd.DataFrame, y: np.ndarray, params: Dict[str, Union[str, int, float]]
    ) -> Dict[str, float]:
        """
        Fit and evaluate a decision tree with given parameters.

        Args:
            X: Feature DataFrame.
            y: Target array.
            params: Dictionary of tree parameters (criterion, max_depth, min_samples_split).

        Returns:
            Dictionary containing training and validation metrics.

        Hint: Follow these steps:
            1. Validate that all required parameters are present in `params`.
            2. Split the data into training and validation sets.
            3. Train a `DecisionTreeClassifier` with the specified parameters.
            4. Evaluate the tree on both training and validation sets.
            5. Collect relevant metrics, including accuracy, depth, and number of leaves.
        """
        # TODO: Implement model fitting and evaluation
        metrics = {
            "train_accuracy": -1,
            "val_accuracy": -1,
            "tree_depth": -1,
            "n_leaves": -1,
            "n_features": -1,
            "n_samples_train": -1,
            "n_samples_val": -1,
        }
        # Step 0: Validate that `params` contains all required keys: "criterion", "max_depth", and "min_samples_split".
        #         Hint: Raise a ValueError if any key is missing.
        # Step 1: Split the data into training and validation sets using `train_test_split`.
        #         Hint: Use stratified splitting to preserve class proportions.
        # Step 2: Create and fit a `DecisionTreeClassifier` using the parameters in `params`.
        #         Hint: Use self.tree to store the trained model.
        # Step 3: Evaluate the model on both training and validation sets.
        #         Hint: Use the `score` method of the classifier for evaluation.
        # Step 4: Collect relevant metrics, including:
        #         - Training and validation accuracies.
        #         - Depth of the tree (use `get_depth()`).
        #         - Number of leaves (use `get_n_leaves()`).
        # Step 5: Replace 'pass' with the correct implementation.
        
        param_keys = params.keys()
        if 'criterion' in param_keys and 'max_depth' in param_keys and 'min_samples_split' in param_keys:
            p = params
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
            self.tree = DecisionTreeClassifier(criterion=p['criterion'], max_depth=p['max_depth'], min_samples_split=p['min_samples_split'])
            self.tree.fit(X_train, y_train)
            metrics['train_accuracy'] = self.tree.score(X_train, y_train)
            metrics['val_accuracy'] = self.tree.score(X_test, y_test)
            metrics['tree_depth'] = self.tree.get_depth()
            metrics['n_leaves'] = self.tree.get_n_leaves()
            metrics['n_features'] = X.shape[1]
            metrics['tree_depth'] = self.tree.get_depth()
            metrics['n_samples_train'] = len(X_train)
            metrics['n_samples_val'] = len(X_test)
            return metrics
        else:
            raise ValueError('key param is missing')


    def visualize_tree(self, feature_names: List[str], class_names: List[str]) -> None:
        """
        Visualize the trained decision tree.

        Args:
            feature_names: List of feature names.
            class_names: List of class names.

        Returns:
            None. Displays a visualization of the decision tree.

        Hint: Use the `plot_tree` function from Scikit-learn to visualize the decision tree.
            1. Ensure the tree is already fitted before attempting visualization.
            2. Set up a plot using Matplotlib for better layout and readability.
            3. Pass appropriate arguments to `plot_tree` to customize the visualization.
        """
        # TODO: Implement decision tree visualization
        # Step 0: Check if the tree has been fitted.
        #         Hint: Raise a ValueError if it is not the case.
        # Step 1: Set up the figure size using Matplotlib.
        # Step 2: Use `plot_tree` from Scikit-learn to visualize the decision tree.
        #         Hint: Customize with `feature_names`, `class_names`, `filled`, and `fontsize` arguments.
        # Step 3: Add a title to the visualization and use `plt.tight_layout()` for better spacing.
        # Step 4: Replace 'pass' with the correct implementation.
        fitted = hasattr(self.tree, 'tree_')
        if fitted :
            fig_size = (20, 20)
            plt.figure(figsize=fig_size)
            plot_tree(self.tree, feature_names=feature_names, class_names=class_names, filled=True, fontsize=10)
            plt.title('Abre de décision')
            plt.show()
        else:
            raise ValueError('tree not fitted')


    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze and return feature importance scores.

        Returns:
            A DataFrame containing:
                - "feature": The name of the feature.
                - "importance_raw": The raw importance scores.
                - "importance_normalized": The normalized importance scores (sum to 1).
                - "rank": Rank of the feature based on normalized importance (1 is highest).
                - "cumulative_importance": Cumulative sum of normalized importance scores.

        Hint: Use the `feature_importances_` attribute of the trained decision tree to extract importance scores.
            1. Normalize the raw scores to ensure they sum to 1.
            2. Create a DataFrame with the above column names.
            3. Sort features by "importance_normalized" in descending order.
            4. Add a "rank" column and compute "cumulative_importance".
        """
        # TODO: Implement feature importance analysis
        # Step 0: Check if the tree has been fitted and feature names are available.
        #         Hint: Raise a ValueError if `self.tree` or `self.feature_names` is None.
        # Step 1: Retrieve raw importance scores using `self.tree.feature_importances_`.
        # Step 2: Normalize the raw scores to ensure they sum to 1.
        # Step 3: Create a DataFrame with columns:
        #         - "feature"
        #         - "importance_raw"
        #         - "importance_normalized"
        #         - "rank"
        #         - "cumulative_importance"
        # Step 4: Sort the DataFrame by "importance_normalized" in descending order.
        # Step 5: Replace 'pass' with the correct implementation.
        fitted = hasattr(self.tree, 'tree_')
        if fitted:
            ft = self.tree.feature_importances_
            df = pd.DataFrame({
                'feature': self.tree.feature_importances_,
                'importance_raw': ft,
                'importance_normalized': ft/ft.sum(),
            })
            df = df.sort_values(by='importance_normalized', ascending=False)
            df["rank"] = df["importance_normalized"].rank(method="dense", ascending=False).astype(int)
            df["cumulative_importance"] = df["importance_normalized"].cumsum()
            return df
        else:
            raise ValueError('not fitted')




    def compare_pruning_methods(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        methods: List[Dict[str, Union[str, int, float]]],
    ) -> pd.DataFrame:
        """
        Compare different pruning methods.

        Args:
            X: Feature DataFrame.
            y: Target array.
            methods: List of parameter dictionaries for different pruning methods. 
                    Each dictionary should include:
                        - "criterion": Splitting criterion ("entropy" or "gini").
                        - "max_depth": Maximum depth of the tree.
                        - "min_samples_split": Minimum number of samples required to split.

        Returns:
            A DataFrame containing metrics for each pruning method. Columns include:
                - "criterion": The splitting criterion used.
                - "max_depth": The maximum depth of the tree.
                - "min_samples_split": The minimum number of samples required to split.
                - "train_accuracy": Accuracy on the training set.
                - "val_accuracy": Accuracy on the validation set.
                - "tree_depth": The depth of the tree.
                - "n_leaves": The number of leaves in the tree.
                - "n_features": The number of features used in training.
                - "n_samples_train": The number of samples in the training set.
                - "n_samples_val": The number of samples in the validation set.
        """
        # TODO: Implement pruning methods comparison
        # Step 1: Initialize an empty list to store results for each method.
        # Step 2: Iterate over the list of pruning methods (`methods`).
        # Step 3: For each method:
        #         - Use `self.fit_and_evaluate` to compute metrics with the provided parameters.
        #         - Create a dictionary with the following keys:
        #           "criterion", "max_depth", "min_samples_split", "train_accuracy", 
        #           "val_accuracy", "tree_depth", "n_leaves", "n_features", 
        #           "n_samples_train", "n_samples_val".
        # Step 4: Append the dictionary to the results list.
        # Step 5: Convert the results list into a DataFrame and return it.
        results = []
        for method in methods:
            metrics = self.fit_and_evaluate(X, y, method)
            result = {
                "criterion": method["criterion"],
                "max_depth": method["max_depth"],
                "min_samples_split": method["min_samples_split"],
                "train_accuracy": metrics["train_accuracy"],
                "val_accuracy": metrics["val_accuracy"],
                "tree_depth": metrics["tree_depth"],
                "n_leaves": metrics["n_leaves"],
                "n_features": X.shape[1],
                "n_samples_train": len(X) - metrics["n_samples_val"],  
                "n_samples_val": metrics["n_samples_val"],
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df



# Example usage:
if __name__ == "__main__":
    # Load sample dataset (you can use any classification dataset)
    from sklearn.datasets import load_iris

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Initialize analyzer
    analyzer = DecisionTreeAnalyzer()

    # Example parameter sets for pruning comparison
    pruning_methods = [
        {"criterion": "gini", "max_depth": 3, "min_samples_split": 2},
        {"criterion": "entropy", "max_depth": 5, "min_samples_split": 5},
        {"criterion": "gini", "max_depth": None, "min_samples_split": 10},
    ]

    # Analyze and compare pruning methods
    results = analyzer.compare_pruning_methods(X, y, pruning_methods)
    print("\nPruning comparison results:")
    print(results)
    analyzer.visualize_tree(feature_names= X.columns, class_names= ['0', '1', '2'])
