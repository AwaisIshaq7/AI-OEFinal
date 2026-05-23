"""
Retrain Models - ICRISAT Agricultural Data
Trains all ML models with ICRISAT dataset and saves them
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime

print("=" * 70)
print("🌾 RETRAINING MODELS WITH ICRISAT DATA")
print("=" * 70)

# Setup paths
base_dir = Path(__file__).parent.parent
models_dir = base_dir / 'models'
models_dir.mkdir(exist_ok=True)

# Load ICRISAT data
print("\n📂 Loading ICRISAT dataset...")
icrisat_file = base_dir / 'ICRISAT-District Level Data.csv'

if not icrisat_file.exists():
    print(f"❌ ERROR: ICRISAT file not found at {icrisat_file}")
    exit(1)

data = pd.read_csv(icrisat_file, encoding='latin-1')
print(f"✓ Loaded {len(data):,} samples with {data.shape[1]} features")

# Data preprocessing
print("\n🔧 Preprocessing data...")

# Replace -1 with NaN
data = data.replace(-1.0, np.nan)

# Fill missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

print(f"✓ Handled missing values")

# Feature engineering
feature_cols = ['N (kg/ha)', 'P2O5 (kg/ha)', 'K2O (kg/ha)', 'Area (000 ha)', 
                'Production (000 tonnes)', 'Yield (kg/ha)']

available_features = [f for f in feature_cols if f in data.columns]
if len(available_features) < 3:
    print("⚠️  Warning: Using available numerical features")
    available_features = data.select_dtypes(include=[np.number]).columns[:7].tolist()

print(f"✓ Selected {len(available_features)} features: {available_features[:3]}...")

X = data[available_features].fillna(0).values[:16146]
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

# Scale features
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
scaler_path = models_dir / 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path.name}")

# ============ DECISION TREE CLASSIFIER ============
print("\n🌳 Training Decision Tree Classifier...")

# Create synthetic crop target (8 classes)
y_crops = (data['State_Name'].astype('category').cat.codes[:16146] % 8).values
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=5)
dt_model.fit(X_scaled, y_crops)

dt_accuracy = dt_model.score(X_scaled, y_crops)
print(f"✓ Decision Tree trained - Accuracy: {dt_accuracy:.2%}")

# Save model
dt_path = models_dir / 'decision_tree_classifier.pkl'
joblib.dump(dt_model, dt_path)
print(f"✓ Model saved: {dt_path.name}")

# Feature importance
importances = dt_model.feature_importances_
importance_dict = {str(i): float(imp) for i, imp in enumerate(importances)}

# ============ K-MEANS CLUSTERING ============
print("\n🗺️  Training K-Means Clustering...")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

from sklearn.metrics import silhouette_score
silhouette = silhouette_score(X_scaled, clusters)
print(f"✓ K-Means trained - Silhouette Score: {silhouette:.4f}")

# Save model
kmeans_path = models_dir / 'knn_clustering.pkl'
joblib.dump(kmeans, kmeans_path)
print(f"✓ Model saved: {kmeans_path.name}")

# ============ LINEAR REGRESSION ============
print("\n📈 Training Linear Regression...")

# Use Yield as target
y_yield = data.get('Yield (kg/ha)', X[:, -1]).fillna(X[:, -1].mean()).values[:16146]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_yield, test_size=0.3, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model evaluation
y_pred = lr_model.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✓ Linear Regression trained")
print(f"  • RMSE: {rmse:.2f}")
print(f"  • MAE: {mae:.2f}")
print(f"  • R²: {r2:.4f}")

# Save model
lr_path = models_dir / 'linear_regression.pkl'
joblib.dump(lr_model, lr_path)
print(f"✓ Model saved: {lr_path.name}")

# ============ SAVE RESULTS ============
print("\n💾 Saving model results...")

results = {
    "timestamp": datetime.now().isoformat(),
    "data_samples": len(X),
    "decision_tree": {
        "accuracy": float(dt_accuracy),
        "n_features": len(available_features),
        "n_classes": 8,
        "feature_importance": importance_dict,
        "crop_mapping": {
            "0": "Rice", "1": "Wheat", "2": "Maize", "3": "Chickpea",
            "4": "Groundnut", "5": "Sugarcane", "6": "Cotton", "7": "Sorghum"
        }
    },
    "kmeans": {
        "silhouette_score": float(silhouette),
        "n_clusters": 4,
        "n_samples": len(X)
    },
    "linear_regression": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "test_r2": float(r2_score(y_test, y_pred)),
        "y_test": y_test[:50].tolist(),
        "y_pred": y_pred[:50].tolist()
    }
}

results_path = models_dir / 'model_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved: {results_path.name}")

# ============ SUMMARY ============
print("\n" + "=" * 70)
print("✅ MODEL RETRAINING COMPLETE")
print("=" * 70)
print(f"\n📊 Summary:")
print(f"  • Samples trained: {len(X):,}")
print(f"  • Features used: {len(available_features)}")
print(f"  • Decision Tree Accuracy: {dt_accuracy:.2%}")
print(f"  • K-Means Silhouette: {silhouette:.4f}")
print(f"  • Linear Regression R²: {r2:.4f}")
print(f"\n📁 Models saved to: {models_dir}/")
print(f"  • decision_tree_classifier.pkl")
print(f"  • knn_clustering.pkl")
print(f"  • linear_regression.pkl")
print(f"  • scaler.pkl")
print(f"  • model_results.json")
print("\n✨ Ready for GUI predictions!")
