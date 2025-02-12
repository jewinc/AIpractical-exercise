from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

STUDENT_LAST_NAME = "P"  # TODO: Fill in your last name
STUDENT_FIRST_NAME = "PH"  # TODO: Fill in your first name

class SVMImplementation:
    """
    A class for implementing and experimenting with Support Vector Machines.
    Includes implementations for different kernels and hyperparameter optimization.
    """

    def __init__(self):
        """Initialize the SVMImplementation class."""
        self.best_model = None
        self.best_params = None

    def preprocess_data(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data using StandardScaler.
        Fits on training data only and transforms both training and validation data.
        
        Args:
            X_train: Training data to fit and transform
            X_val: Optional validation data to transform
            
        Returns:
            Tuple of transformed training data and (if provided) validation data
        """
        scaler = StandardScaler()
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
            
        # Fit on training data only
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None:
            if len(X_val.shape) == 1:
                X_val = X_val.reshape(-1, 1)
            # Transform validation data using training data statistics
            X_val_scaled = scaler.transform(X_val)
            return X_train_scaled, X_val_scaled
            
        return X_train_scaled

    def train_linear_svm(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Dict:
        """
        Train a linear SVM classifier and return performance metrics.
        """
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess data properly
        X_train_scaled, X_val_scaled = self.preprocess_data(X_train, X_val)

        # Train linear SVM
        svm = SVC(kernel="linear", C=C, random_state=42)
        svm.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = svm.predict(X_val_scaled)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "n_support_vectors": len(svm.support_vectors_),
        }

        return metrics

    def train_kernel_svm(
        self, X: np.ndarray, y: np.ndarray, kernel: str, **kernel_params
    ) -> Dict:
        """
        Train an SVM with the specified kernel and return performance metrics.
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocess data properly
        X_train_scaled, X_val_scaled = self.preprocess_data(X_train, X_val)

        # Create and train SVM with specified kernel
        svm = SVC(kernel=kernel, random_state=42, **kernel_params)
        svm.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = svm.predict(X_val_scaled)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "n_support_vectors": len(svm.support_vectors_),
        }

        return metrics

    def parameter_optimization(
        self, X: np.ndarray, y: np.ndarray, kernel: str
    ) -> Tuple[Dict, float]:
        """
        Perform grid search to find optimal SVM parameters.
        """
        # Define parameter grid based on kernel type
        if kernel == "linear":
            param_grid = {"C": [0.1, 1, 10, 100]}
        elif kernel == "rbf":
            param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.1, 1]}
        elif kernel == "poly":
            param_grid = {
                "C": [0.1, 1, 10],
                "degree": [2, 3, 4],
                "gamma": ["scale", "auto"],
            }
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

        # Create SVM and GridSearchCV objects
        svm = SVC(kernel=kernel, random_state=42)
        
        # Create a pipeline that includes preprocessing
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', svm)
        ])
        
        # Perform grid search with proper cross-validation
        grid_search = GridSearchCV(pipeline, {'svm__' + k: v for k, v in param_grid.items()}, 
                                 cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X, y)

        # Extract SVM-specific parameters from the best parameters
        best_svm_params = {k.replace('svm__', ''): v 
                          for k, v in grid_search.best_params_.items()}

        return best_svm_params, grid_search.best_score_

    def compare_kernels(
        self, X: np.ndarray, y: np.ndarray, kernels: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare performance of different kernel functions.
        """
        results = {}

        for kernel in kernels:
            # Optimize parameters for each kernel
            best_params, _ = self.parameter_optimization(X, y, kernel)

            # Train and evaluate model with best parameters
            metrics = self.train_kernel_svm(X, y, kernel, **best_params)
            results[kernel] = metrics

        return results

    def analyze_support_vectors(
        self, X: np.ndarray, y: np.ndarray, kernel: str
    ) -> Dict:
        """
        Analyze the support vectors for a given kernel.
        """
        # Preprocess all data
        X_scaled = self.preprocess_data(X)
        
        # Train SVM
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_scaled, y)

        # Get support vectors and their indices
        support_vectors = svm.support_vectors_
        support_vector_indices = svm.support_

        # Calculate distances to decision boundary for support vectors
        decision_function_vals = abs(svm.decision_function(support_vectors))

        analysis = {
            "n_support_vectors": len(support_vectors),
            "support_vector_ratio": len(support_vectors) / len(X),
            "avg_distance_to_boundary": np.mean(decision_function_vals),
            "std_distance_to_boundary": np.std(decision_function_vals),
            "n_support_vectors_per_class": svm.n_support_.tolist(),
        }

        return analysis


if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    
    # Initialize implementation
    svm_impl = SVMImplementation()

    # Compare different kernels
    kernel_results = svm_impl.compare_kernels(
        X, y, kernels=["linear", "rbf", "poly"]
    )

    print("Kernel Comparison Results:")
    for kernel, metrics in kernel_results.items():
        print(f"\n{kernel} kernel:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
