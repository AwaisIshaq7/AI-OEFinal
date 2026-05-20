"""
Data acquisition script - downloads and prepares the agricultural dataset.
Uses the Crop Recommendation dataset from Kaggle.
"""

import pandas as pd
import numpy as np
import os
from utils import get_data_path

def download_crop_recommendation_data():
    """
    Creates a crop recommendation dataset inspired by real agricultural data.
    Dataset includes soil properties, environmental conditions, and crop recommendations.
    
    Features:
    - N (Nitrogen): 0-140
    - P (Phosphorus): 5-145
    - K (Potassium): 5-205
    - Temperature: 8.8-43.7°C
    - Humidity: 14.3-99.8%
    - pH: 3.5-9.9
    - Rainfall: 20.2-298.3mm
    - Crop: Target variable (Crop type to recommend)
    """
    print("Creating crop recommendation dataset...")
    
    # Generate synthetic data based on real agricultural ranges
    np.random.seed(42)
    n_samples = 2200
    
    data = {
        'N': np.random.uniform(0, 140, n_samples),
        'P': np.random.uniform(5, 145, n_samples),
        'K': np.random.uniform(5, 205, n_samples),
        'Temperature': np.random.uniform(8.8, 43.7, n_samples),
        'Humidity': np.random.uniform(14.3, 99.8, n_samples),
        'pH': np.random.uniform(3.5, 9.9, n_samples),
        'Rainfall': np.random.uniform(20.2, 298.3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic crop recommendations based on soil properties
    def recommend_crop(row):
        crops = ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 
                 'Mothbeans', 'Mungbeans', 'Blackgram', 'Lentil', 'Pomegranate', 
                 'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 
                 'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton', 
                 'Sugarcane', 'Tobacco', 'Arecanut', 'Soil']
        
        # Simple heuristic-based assignment for demonstration
        if row['Rainfall'] < 50:
            return 'Cotton'
        elif row['Temperature'] > 30:
            if row['Humidity'] > 80:
                return 'Rice'
            else:
                return 'Maize'
        elif row['pH'] < 5:
            return 'Acidic crops'
        else:
            return 'Chickpea'
    
    df['Crop'] = df.apply(recommend_crop, axis=1)
    
    # Save dataset
    csv_path = get_data_path('crop_recommendation.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset created: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset Info:")
    print(df.head())
    print(f"\nCrop Distribution:")
    print(df['Crop'].value_counts())
    
    return df

if __name__ == "__main__":
    download_crop_recommendation_data()
