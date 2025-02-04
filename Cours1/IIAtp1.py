from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Cheng"  
STUDENT_FIRST_NAME = "Jewin"  


class DataPreprocessor:
    """
    A class for preprocessing data with basic cleaning and scaling operations.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Strategy:
        - Numerical columns: fill with median
        - Categorical columns: fill with mode

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        # Example Input:
        #    A    B
        # 0  1.0  NaN
        # 1  NaN  3.0
        # Example Output:
        #    A    B
        # 0  1.0  3.0
        # 1  1.0  3.0
        # Edge Cases to Consider:
        # - Numerical column with all NaN values: Should fill with 0 or exclude column.
        # - Categorical column with all NaN values: Should fill with 'unknown'.
        # TODO: Implement missing value handling
        
        #Itérer sur les colonnes, selon le type de la colonne faire un fillna median/mode avec df.select_dtypes(include=[types]).columns et fillna(value)
        val_age: float = df['age'].median()
        val_salary: float = df['salary'].median()
        val_exp: float = df['experience'].median()
        val_department: str = df['department'].mode().iloc[0] if not df['department'].mode().empty else 'IT'
        val_educ: str = df['education'].mode()
        return pd.DataFrame({
            'age': df['age'].fillna(val_age),
            'salary': df['salary'].fillna(val_salary),
            'experience': df['experience'].fillna(val_exp),
            'department': df['department'].fillna(val_department).astype(str),
            'education': df['education'].fillna(val_educ),
        })

    def remove_outliers(
        self, df: pd.DataFrame, columns: list, threshold: float = 3
    ) -> pd.DataFrame:
        """
        Remove outliers from specified numerical columns using z-score method.

        Args:
            df: Input DataFrame
            columns: List of numerical columns to check for outliers
            threshold: Z-score threshold (default = 3)

        Returns:
            DataFrame with outliers removed
        """
        # Hint:
        # 1. For each column in 'columns', calculate the mean and standard deviation.
        # 2. Compute the z-score for each value in the column.
        # 3. Drop rows where the z-score exceeds 'threshold'.
        
        # Z-score calculation (pseudo-code):
        # z_scores = (column_values - column_mean) / column_std
        # Use 'np.abs(z_scores) < threshold' to filter rows.

        # TODO: Implement outlier removal using z-score method
        tmp: 'boolean conditional Series to select rows' = pd.Series(np.full(df.shape[0], True))
        for col in columns:
            col_mean: float = df[col].mean()
            col_std: float = df[col].std()
            z_scores: float = (df[col] - col_mean) / col_std
            tmp = tmp & (np.abs(z_scores) < threshold)
        return df[tmp]
        
        

    def scale_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale

        Returns:
            DataFrame with scaled features
        """
        # Tip: After scaling, check if the column's mean is close to 0 and the standard deviation is close to 1.
        # Example:
        # print(df[columns].mean())  # Should be near 0
        # print(df[columns].std())   # Should be near 1
        res: pd.DataFrame = pd.DataFrame(self.scaler.fit_transform(df[columns]), columns=columns)
        return pd.concat([res, df[['department', 'education']]], axis=1)

    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        First fills NaN values with a placeholder to ensure consistent encoding.

        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode

        Returns:
            DataFrame with encoded categorical variables
        """
        
        return pd.get_dummies(df, columns=columns, dtype='uint8')

    def preprocess_data(
        self, df: pd.DataFrame, numerical_columns: list, categorical_columns: list
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            numerical_columns: List of numerical columns
            categorical_columns: List of categorical columns

        Returns:
            Tuple containing:
            - Preprocessed DataFrame
            - Dictionary with preprocessing statistics
        """
        # Preprocessing Steps:
        # 1. Handle missing values (use 'handle_missing_values').
        # 2. Remove outliers (use 'remove_outliers').
        # 3. Scale features (use 'scale_features').
        # 4. Encode categorical variables (use 'encode_categorical').
        # Ensure each step is applied in the specified order.
        # TODO: Implement full preprocessing pipeline
        # Should return (preprocessed_df, stats_dict)
        preprocessed_df: pd.DataFrame = self.encode_categorical(
            df= self.scale_features(
                df = self.remove_outliers(
                    df= self.handle_missing_values(df),
                    columns= numerical_columns
                ).reset_index(drop=True),
                columns=numerical_columns
            ),
            columns=categorical_columns
        )
        stats_dict: dict = {
            'age_mean': preprocessed_df['age'].mean(),
            'age_std': preprocessed_df['age'].std(),
            'salary_mean': preprocessed_df['salary'].mean(),
            'salary_std': preprocessed_df['salary'].std(),
            'experience_mean': preprocessed_df['experience'].mean(),
            'experience_std': preprocessed_df['experience'].std(),
            'num_cols': preprocessed_df.shape[1]
        }
        return (preprocessed_df, stats_dict)



# Example usage:
if __name__ == "__main__":
    # Load sample dataset
    df = pd.read_csv("sample_dataset.csv")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Define columns
    numerical_cols = ["age", "salary", "experience"]
    categorical_cols = ["department", "education"]

    # Preprocess data
    processed_df, stats = preprocessor.preprocess_data(
        df, numerical_cols, categorical_cols
    )

    print("Preprocessing complete!")
    print("\nPreprocessing statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
