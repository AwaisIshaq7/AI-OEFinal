"""
Data preprocessing module for agricultural dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
from utils import get_data_path

class DataPreprocessor:
    """Handles data loading, cleaning, and transformation."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load dataset from CSV."""
        self.data = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded: {self.data.shape}")
        return self.data
    
    def handle_missing_values(self, strategy='mean'):
        """Impute missing values."""
        imputer = SimpleImputer(strategy=strategy)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        print(f"Missing values handled using {strategy} strategy")
        return self.data
    
    def handle_outliers(self, columns=None, method='iqr'):
        """Remove outliers using IQR method."""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & 
                                      (self.data[col] <= upper_bound)]
        
        print(f"Outliers removed. Data shape: {self.data.shape}")
        return self.data
    
    def encode_categorical(self, columns=None):
        """Encode categorical variables."""
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Categorical encoding completed for {len(columns)} columns")
        return self.data
    
    def scale_features(self, columns=None, exclude_cols=None):
        """Scale numeric features to standard distribution."""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Exclude target columns from scaling
        if exclude_cols is not None:
            columns = [col for col in columns if col not in exclude_cols]
        
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        print("Features scaled successfully")
        return self.data
    
    def get_feature_importance_columns(self):
        """Return list of feature columns (excluding target)."""
        # This will be customized based on actual dataset
        return self.data.columns.tolist()
    
    def preprocess_pipeline(self, target_column=None):
        """Execute complete preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        self.load_data()
        self.handle_missing_values(strategy='mean')
        self.handle_outliers()
        self.encode_categorical()
        self.scale_features()
        print("Preprocessing pipeline completed!")
        return self.data
