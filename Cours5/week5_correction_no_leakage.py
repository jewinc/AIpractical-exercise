from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump, load

STUDENT_LAST_NAME = "P"  # TODO: Fill in your last name
STUDENT_FIRST_NAME = "PH"  # TODO: Fill in your first name

class SVMImplementation:
    """ A class for implementing Support Vector Machines with proper validation and no data leakage. """

    def __init__(self):
        self.best_pipeline = None
        self.best_params = None
        self.best_val_score = None

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Validate input arrays. """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input arrays cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if not np.isfinite(X).all():
            raise ValueError("Input features contain non-finite values")

    def parameter_optimization(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray,
        y_val: np.ndarray,
        kernel: str, 
        n_jobs: Optional[int] = None
    ) -> Tuple[Dict, Pipeline, float]:
        """ 
        Perform parameter optimization using training data and validate on a separate validation set.
        Returns the best parameters, trained pipeline, and validation score.
        """
        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)

        # Define parameter grid based on kernel type
        param_grid = {
            "svm__C": [0.1, 1, 10, 100]
        }
        if kernel == "rbf":
            param_grid["svm__gamma"] = ["scale", "auto", 0.1, 1]
        elif kernel == "poly":
            param_grid["svm__degree"] = [2, 3, 4]
            param_grid["svm__gamma"] = ["scale", "auto"]

        # Create a pipeline with scaling inside
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, random_state=42))
        ])

        # Perform grid search with 5-fold cross-validation on training data
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=n_jobs
        )
        grid_search.fit(X_train, y_train)

        # Evaluate best model on validation set
        val_score = grid_search.score(X_val, y_val)

        # Save best pipeline and parameters
        best_pipeline = grid_search.best_estimator_
        best_params = {k.replace("svm__", ""): v for k, v in grid_search.best_params_.items()}

        return best_params, best_pipeline, val_score

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """ Evaluate the best model on the isolated test set. """
        if self.best_pipeline is None:
            raise ValueError("No trained model available for evaluation")

        y_pred = self.best_pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        return metrics

    def compare_kernels(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        kernels: List[str]
    ) -> Dict[str, Dict]:
        """ Compare SVM models using different kernels with proper validation. """
        self._validate_inputs(X_train, y_train)
        self._validate_inputs(X_val, y_val)
        self._validate_inputs(X_test, y_test)

        results = {}

        for kernel in kernels:
            print(f"Optimizing SVM with {kernel} kernel...")

            # Optimize parameters using training data and validate on validation set
            best_params, trained_pipeline, val_score = self.parameter_optimization(
                X_train, y_train, X_val, y_val, kernel
            )

            # **FIX:** Set self.best_pipeline before evaluation
            self.best_pipeline = trained_pipeline  

            # Evaluate on the test set (which was never used in training or validation)
            test_metrics = self.evaluate_model(X_test, y_test)

            # Store both validation and test metrics
            results[kernel] = {
                "validation_accuracy": val_score,
                "test_metrics": test_metrics,
                "best_pipeline": trained_pipeline  # Store per-kernel pipeline
            }

        return results


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    svm_impl = SVMImplementation()
    kernel_results = svm_impl.compare_kernels(X_train, y_train, X_val, y_val, X_test, y_test, ["linear", "rbf", "poly"])


    best_kernel = max(kernel_results, key=lambda k: kernel_results[k]['test_metrics']['accuracy'])
    print(f"\nBest kernel based on test accuracy: {best_kernel}")
    # print accuracy of the best model
    print(f"Best model test accuracy: {kernel_results[best_kernel]['test_metrics']['accuracy']}")

    
    # Save the best model
    dump(kernel_results[best_kernel]["best_pipeline"], f"{STUDENT_LAST_NAME}_{STUDENT_FIRST_NAME}_best_model.joblib")
    # load the best model
    loaded_pipeline = load(f"{STUDENT_LAST_NAME}_{STUDENT_FIRST_NAME}_best_model.joblib")
    # Evaluate the loaded model loaded_pipeline on the test set
    y_pred = loaded_pipeline.predict(X_test)
    print(f"Loaded model test accuracy: {accuracy_score(y_test, y_pred)}")