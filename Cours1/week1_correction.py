import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
#! DO NOT ADD OR MODIFY IMPORTS - YOU NEED TO WORK WITH THE ABOVE IMPORTS !

STUDENT_LAST_NAME = "Paris"  # TODO: Fill in your last name
STUDENT_FIRST_NAME = "PH"  # TODO: Fill in your first name

class DataPreprocessor:
    """
    A class for preprocessing data with basic cleaning and scaling operations.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        df = df.copy()
        
        # Handle numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())
            
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df
        
    def remove_outliers(self, df: pd.DataFrame, columns: list, threshold: float = 3) -> pd.DataFrame:
        """
        Remove outliers from specified numerical columns using z-score method.
        """
        df = df.copy()
        
        for column in columns:
            if df[column].dtype in ['int64', 'float64']:
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                df = df[z_scores < threshold]
                
        return df
        
    def scale_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        """
        df = df.copy()
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
        
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        """
        df = df.copy()
        for column in columns:
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
        return df
        
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       numerical_columns: list,
                       categorical_columns: list) -> Tuple[pd.DataFrame, dict]:
        """
        Complete preprocessing pipeline.
        """
        # Initialize stats dictionary
        stats = {
            'initial_rows': len(df),
            'initial_missing': df.isnull().sum().sum(),
            'categorical_columns': len(categorical_columns),
            'numerical_columns': len(numerical_columns)
        }
        
        # Apply preprocessing steps
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df, numerical_columns)
        df = self.scale_features(df, numerical_columns)
        df = self.encode_categorical(df, categorical_columns)
        
        # Update stats
        stats.update({
            'final_rows': len(df),
            'final_columns': len(df.columns),
            'outliers_removed': stats['initial_rows'] - len(df),
            'final_missing': df.isnull().sum().sum()
        })
        
        return df, stats

if __name__ == "__main__":
    # Load sample dataset
    df = pd.read_csv("week1/lab/sample_dataset.csv")

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