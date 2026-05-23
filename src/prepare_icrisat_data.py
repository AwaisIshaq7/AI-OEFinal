"""
ICRISAT Data Preparation and Model Retraining Script
Processes real agricultural data from ICRISAT for model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

class ICRISATDataPreprocessor:
    """Prepare ICRISAT data for ML models"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.processed_data = None
        
    def load_data(self):
        """Load ICRISAT CSV file"""
        print(f"📂 Loading ICRISAT data from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.df)} rows × {len(self.df.columns)} columns")
        print(f"📊 Columns: {list(self.df.columns[:10])}...")
        return self
    
    def handle_missing_values(self):
        """Handle -1 values (missing data indicators)"""
        print("\n🔧 Handling missing values...")
        
        # Replace -1 with NaN
        self.df = self.df.replace(-1.0, np.nan)
        
        # For numeric columns, fill NaN with mean
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isna().sum() > 0:
                mean_val = self.df[col].mean()
                self.df[col].fillna(mean_val, inplace=True)
                print(f"  ✓ {col}: filled {self.df[col].isna().sum()} NaNs with mean={mean_val:.2f}")
        
        print(f"✅ No missing values remaining: {self.df.isna().sum().sum() == 0}")
        return self
    
    def extract_features(self):
        """Extract key agricultural features"""
        print("\n📊 Extracting features...")
        
        # Key features from ICRISAT data
        feature_cols = [
            'RICE AREA (1000 ha)', 'RICE YIELD (Kg per ha)',
            'WHEAT AREA (1000 ha)', 'WHEAT YIELD (Kg per ha)',
            'MAIZE AREA (1000 ha)', 'MAIZE YIELD (Kg per ha)',
            'CHICKPEA AREA (1000 ha)', 'CHICKPEA YIELD (Kg per ha)',
            'GROUNDNUT AREA (1000 ha)', 'GROUNDNUT YIELD (Kg per ha)',
            'SUGARCANE AREA (1000 ha)', 'SUGARCANE YIELD (Kg per ha)',
            'COTTON AREA (1000 ha)', 'COTTON YIELD (Kg per ha)',
            'SORGHUM AREA (1000 ha)', 'SORGHUM YIELD (Kg per ha)',
        ]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in self.df.columns]
        print(f"✓ Using {len(available_features)} features")
        
        # Also include year and basic location info for context
        context_cols = ['Year', 'Dist Name']
        
        return available_features, context_cols
    
    def create_synthetic_soil_climate(self):
        """Generate synthetic but realistic soil and climate features"""
        print("\n🌡️  Creating synthetic soil/climate features...")
        
        np.random.seed(42)
        n_samples = len(self.df)
        
        # Generate realistic ranges based on Indian agriculture
        data = {
            'N': np.random.uniform(40, 100, n_samples),              # Nitrogen (kg/ha)
            'P': np.random.uniform(10, 50, n_samples),               # Phosphorus (kg/ha)
            'K': np.random.uniform(30, 80, n_samples),               # Potassium (kg/ha)
            'Temperature': np.random.uniform(15, 35, n_samples),     # °C
            'Humidity': np.random.uniform(50, 90, n_samples),        # %
            'pH': np.random.uniform(6.0, 8.0, n_samples),            # soil pH
            'Rainfall': np.random.uniform(500, 2000, n_samples),     # mm
        }
        
        synthetic_df = pd.DataFrame(data)
        print(f"✅ Generated 7 synthetic features for {n_samples} samples")
        return synthetic_df
    
    def prepare_for_classification(self, features_df, synthetic_df):
        """Prepare data for decision tree classification"""
        print("\n🌳 Preparing classification data...")
        
        # Create target: recommend crop based on yield
        yields = features_df[[col for col in features_df.columns if 'YIELD' in col]].values
        
        # Determine best crop (class)
        best_crop_idx = np.argmax(yields, axis=1)
        
        # Map to crop names
        crop_names = ['Rice', 'Wheat', 'Maize', 'Chickpea', 'Groundnut', 'Sugarcane', 'Cotton', 'Sorghum']
        y_class = np.array([crop_names[min(i, len(crop_names)-1)] for i in best_crop_idx])
        
        # Encode crop names
        unique_crops = np.unique(y_class)
        crop_encoding = {crop: idx for idx, crop in enumerate(unique_crops)}
        y_encoded = np.array([crop_encoding[crop] for crop in y_class])
        
        print(f"✓ Created {len(unique_crops)} crop classes: {list(unique_crops)}")
        print(f"✓ Class distribution:\n{pd.Series(y_class).value_counts()}")
        
        return synthetic_df, y_encoded, crop_encoding
    
    def prepare_for_clustering(self, features_df, synthetic_df):
        """Prepare data for K-means clustering"""
        print("\n🗺️  Preparing clustering data...")
        
        # Use synthetic features for clustering zones
        X_cluster = synthetic_df.copy()
        
        print(f"✓ Clustering data shape: {X_cluster.shape}")
        return X_cluster
    
    def prepare_for_regression(self, features_df, synthetic_df):
        """Prepare data for linear regression (yield prediction)"""
        print("\n📈 Preparing regression data...")
        
        # Target: average yield across crops
        yields = features_df[[col for col in features_df.columns if 'YIELD' in col]].values
        y_yield = np.mean(yields, axis=1)
        
        print(f"✓ Target yield range: {y_yield.min():.2f} - {y_yield.max():.2f} kg/ha")
        print(f"✓ Mean yield: {y_yield.mean():.2f} kg/ha")
        
        return synthetic_df, y_yield
    
    def prepare_all_data(self):
        """Complete preparation pipeline"""
        print("=" * 60)
        print("🚀 ICRISAT DATA PREPARATION PIPELINE")
        print("=" * 60)
        
        # Load and clean
        self.load_data()
        self.handle_missing_values()
        
        # Extract features
        available_features, context_cols = self.extract_features()
        features_df = self.df[available_features].copy()
        
        # Create synthetic features
        synthetic_df = self.create_synthetic_soil_climate()
        
        # Prepare for each model
        X_class, y_class, crop_encoding = self.prepare_for_classification(features_df, synthetic_df)
        X_cluster = self.prepare_for_clustering(features_df, synthetic_df)
        X_reg, y_reg = self.prepare_for_regression(features_df, synthetic_df)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_class)
        
        print("\n" + "=" * 60)
        print("📊 FINAL DATA SUMMARY")
        print("=" * 60)
        print(f"✓ Classification: X={X_scaled.shape}, y={y_class.shape}")
        print(f"✓ Clustering: X={X_cluster.shape}")
        print(f"✓ Regression: X={X_reg.shape}, y={y_reg.shape}")
        print(f"✓ Crop classes: {len(crop_encoding)}")
        
        return {
            'X_class': X_scaled,
            'y_class': y_class,
            'crop_encoding': crop_encoding,
            'X_cluster': X_cluster,
            'X_reg': X_reg,
            'y_reg': y_reg,
            'scaler': scaler,
            'feature_names': list(X_class.columns) if hasattr(X_class, 'columns') else 
                            ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        }


def prepare_icrisat_data(csv_path):
    """Main function to prepare ICRISAT data"""
    preprocessor = ICRISATDataPreprocessor(csv_path)
    data = preprocessor.prepare_all_data()
    return data, preprocessor.df


if __name__ == "__main__":
    csv_path = Path("d:/AIOEL/ICRISAT-District Level Data.csv")
    if csv_path.exists():
        data, df = prepare_icrisat_data(str(csv_path))
        print(f"\n✅ Data prepared successfully!")
        print(f"📁 Save this data for model training")
    else:
        print(f"❌ File not found: {csv_path}")
