from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Cheng"  # TODO: Fill in your last name
STUDENT_FIRST_NAME = "Jewin"  # TODO: Fill in your first name

class SVMImplementation:
    """
    A class for implementing and experimenting with Support Vector Machines.
    Includes implementations for different kernels and hyperparameter optimization.
    """

    def __init__(self):
        """Initialize the SVMImplementation class with necessary preprocessing tools."""
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = None

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data using StandardScaler.

        Args:
            X: Input features

        Returns:
            Scaled features
        """
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler.transform(X)


    def train_linear_svm(
            self, 
            X: np.ndarray,
            y: np.ndarray,
            C: float = 1.0
        ) -> Dict:
        """
        Train a linear SVM classifier and return performance metrics.

        Args:
            X_train: Training features
            y_train: Training labels
            
            en principe:
            x_test: Testset features
            y_test: Testset labelsC: Regularization parameter

        Returns:
            Dictionary containing model performance metrics
        """
        metrics = {
            "accuracy": -1,
            "precision": -1,
            "recall": -1,
            "f1": -1,
            "n_support_vectors": -1,
        }
        model = SVC(C=C, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        
        metrics['accuracy'] = accuracy_score(y, pred)
        metrics['precision'] = precision_score(y, pred)
        metrics['recall'] = recall_score(y, pred)
        metrics['f1'] = f1_score(y, pred)
        metrics['n_support_vectors'] = len(model.support_)
        return metrics

    def train_kernel_svm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel: str, **kernel_params
    ) -> Dict:
        """
        Train an SVM with the specified kernel and return performance metrics.

        Args:
            X: Training features
            y: Training labels
            kernel: Kernel type ('rbf', 'poly', or 'sigmoid')
            **kernel_params: Additional kernel parameters

        Returns:
            Dictionary containing model performance metrics
        """
        # TODO: Implement kernel SVM training and evaluation
        flag = False
        if not(kernel_params=={}):
            best_params = kernel_params['kernel_params']
            kernel = best_params['kernel']
            flag = True

        
        metrics = {
            "accuracy": -1,
            "precision": -1,
            "recall": -1,
            "f1": -1,
            "n_support_vectors": -1,
        }
        if kernel=='precomputed':
            X = self.custom_kernel(X, X)

        if flag:
            model = SVC(kernel=kernel, C=best_params['C'], gamma=best_params['gamma'], random_state=42)
        else:
            model = SVC(kernel=kernel, random_state=42)

        model.fit(X, y)
        pred = model.predict(X)
        metrics['accuracy'] = accuracy_score(y, pred)
        metrics['precision'] = precision_score(y, pred)
        metrics['recall'] = recall_score(y, pred)
        metrics['f1'] = f1_score(y, pred)
        metrics['n_support_vectors'] = len(model.support_)
        return metrics

    def custom_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Implement a custom kernel function.
        This should be a combination of multiple basic kernels.

        Args:
            X1: First array of features
            X2: Second array of features

        Returns:
            Kernel matrix
        """
        f1 = np.dot(X1, X2.T)
        f2 = (0.1 * f1 + 1)**3
        return (0.2 * f1 + 0.8 * f2)


    def parameter_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel: str
    ) -> Tuple[Dict, float]:
        """
        Perform grid search to find optimal SVM parameters.

        Args:
            X: Training features
            y: Training labels
            kernel: Kernel type to optimize

        Returns:
            Tuple containing best parameters and best score
        """
        hparams: Dict = {
            'C': [0.1, 1, 10], 
            'gamma': [1, 0.1, 0.01],
            'kernel': [kernel],
            'random_state': [42]
        }
        search = GridSearchCV(estimator=SVC(), param_grid=hparams, scoring='f1', return_train_score=True, cv=5)
        search.fit(X, y)
        self.best_model = search.best_estimator_
        self.best_params = {
            'gamma': self.best_model.gamma,
            'C': self.best_model.C,
            'kernel': kernel,
            'random_state': 42
        }
        return (self.best_params, search.best_score_)


    def compare_kernels(
        self, 
        X: np.ndarray,
        y: np.ndarray,
        kernels: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare performance of different kernel functions.

        Args:
            X: Training features
            y: Training labels
            kernels: List of kernel types to compare

        Returns:
            Dictionary containing performance metrics for each kernel
        """
        metrics = {}
        for kernel in kernels:
            best_params = self.parameter_optimization(X, y, kernel=kernel)[0]
            perf = self.train_kernel_svm(X, y, kernel, kernel_params=best_params) #, x_test, y_test, kernel)
            metrics[kernel] = perf
        return metrics

    def analyze_support_vectors(
        self, X: np.ndarray, y: np.ndarray, kernel: str
    ) -> Dict:
        """
        Analyze the support vectors for a given kernel.

        Args:
            X: Training features
            y: Training labels
            kernel: Kernel type to analyze

        Returns:
            Dictionary containing support vector analysis
        """
        # TODO: Implement support vector analysis
        analysis = {
            "n_support_vectors": -1,
            "support_vector_ratio": -1,
            "avg_distance_to_boundary": -1,
            "std_distance_to_boundary": -1,
            "n_support_vectors_per_class": None, # List of support vectors per class
        }
        model = SVC(kernel=kernel)
        model.fit(X, y)
        decision_function = model.decision_function(X)
        distances = np.abs(decision_function[model.support_])

        analysis['n_support_vectors'] = len(model.support_vectors_)
        analysis['support_vector_ratio'] = len(model.support_vectors_) / len(X)
        analysis['avg_distance_to_boundary'] = np.mean(distances)
        analysis['std_distance_to_boundary'] = np.std(distances)
        analysis['n_support_vectors_per_class'] = model.n_support_.tolist()
        return analysis


# Example usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and use the implementation
    svm_impl = SVMImplementation()

    # Example workflow
    X_train_scaled = svm_impl.preprocess_data(X_train)
    X_test_scaled = svm_impl.preprocess_data(X_test)
    
    # Compare different kernels
    kernel_results = svm_impl.compare_kernels(
        X_train_scaled,
        y_train,
        #x_test=X_test_scaled,
        #y_test=y_test,
        kernels=["linear", "rbf", "poly"]
    )

    print("Kernel Comparison Results:")
    for kernel, metrics in kernel_results.items():
        print(f"\n{kernel} kernel:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")