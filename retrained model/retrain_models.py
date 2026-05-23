"""
Retrain ML Models with ICRISAT Real Agricultural Data
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, mean_squared_error, r2_score, mean_absolute_error
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from prepare_icrisat_data import prepare_icrisat_data


def retrain_all_models(csv_path, save_dir='models'):
    """Retrain all three models with ICRISAT data"""
    
    print("\n" + "=" * 70)
    print("🔄 RETRAINING MODELS WITH ICRISAT DATA")
    print("=" * 70)
    
    # Prepare data
    print("\n📊 Step 1: Preparing ICRISAT data...")
    data, df = prepare_icrisat_data(csv_path)
    
    # Extract components
    X_class = data['X_class']
    y_class = data['y_class']
    crop_encoding = data['crop_encoding']
    X_cluster = data['X_cluster']
    X_reg = data['X_reg']
    y_reg = data['y_reg']
    scaler = data['scaler']
    
    results = {}
    
    # ============================================
    # 1. DECISION TREE CLASSIFIER (Crop Recommendation)
    # ============================================
    print("\n" + "-" * 70)
    print("🌳 Training Decision Tree Classifier...")
    print("-" * 70)
    
    # Split data
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    # Train
    dt_classifier = DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    dt_classifier.fit(X_train_class, y_train_class)
    
    # Evaluate
    y_pred_class = dt_classifier.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    
    print(f"✅ Decision Tree trained!")
    print(f"   • Train samples: {len(X_train_class)}")
    print(f"   • Test samples: {len(X_test_class)}")
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Classes: {len(np.unique(y_class))}")
    
    # Feature importance
    feature_importance = dt_classifier.feature_importances_
    feature_names = data['feature_names']
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importance)}
    print(f"   • Top 3 features: {sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Save
    joblib.dump(dt_classifier, Path(save_dir) / 'decision_tree_classifier.pkl')
    results['decision_tree'] = {
        'accuracy': accuracy,
        'feature_importance': importance_dict,
        'n_features': len(feature_names),
        'n_classes': len(np.unique(y_class)),
        'crop_mapping': {v: k for k, v in crop_encoding.items()}
    }
    
    # ============================================
    # 2. K-MEANS CLUSTERING (Crop Zone Analysis)
    # ============================================
    print("\n" + "-" * 70)
    print("🗺️  Training K-Means Clustering...")
    print("-" * 70)
    
    # Train
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    clusters = kmeans.fit_predict(X_cluster)
    
    # Evaluate
    silhouette = silhouette_score(X_cluster, clusters)
    
    print(f"✅ K-Means trained!")
    print(f"   • Data points: {len(X_cluster)}")
    print(f"   • Clusters: {4}")
    print(f"   • Silhouette Score: {silhouette:.4f}")
    print(f"   • Inertia: {kmeans.inertia_:.2f}")
    
    # Cluster distribution
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"   • Distribution: {dict(zip(unique, counts))}")
    
    # Save
    joblib.dump(kmeans, Path(save_dir) / 'knn_clustering.pkl')
    results['kmeans'] = {
        'n_clusters': 4,
        'silhouette_score': silhouette,
        'inertia': float(kmeans.inertia_),
        'cluster_sizes': {int(c): int(cnt) for c, cnt in zip(unique, counts)}
    }
    
    # ============================================
    # 3. LINEAR REGRESSION (Yield Prediction)
    # ============================================
    print("\n" + "-" * 70)
    print("📈 Training Linear Regression...")
    print("-" * 70)
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Train
    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    
    # Evaluate
    y_pred_reg = lr.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"✅ Linear Regression trained!")
    print(f"   • Train samples: {len(X_train_reg)}")
    print(f"   • Test samples: {len(X_test_reg)}")
    print(f"   • RMSE: {rmse:.4f} kg/ha")
    print(f"   • MAE: {mae:.4f} kg/ha")
    print(f"   • R² Score: {r2:.4f}")
    print(f"   • Coefficients: {list(zip(feature_names, lr.coef_))[:3]}")
    
    # Save
    joblib.dump(lr, Path(save_dir) / 'linear_regression.pkl')
    results['linear_regression'] = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_features': len(feature_names),
        'y_test': y_test_reg.tolist(),
        'y_pred': y_pred_reg.tolist(),
        'test_r2': r2
    }
    
    # ============================================
    # Save Results
    # ============================================
    print("\n" + "-" * 70)
    print("💾 Saving results...")
    print("-" * 70)
    
    # Save model results
    results_file = Path(save_dir) / 'model_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved model results to: {results_file}")
    
    # Save scaler for GUI
    joblib.dump(scaler, Path(save_dir) / 'scaler.pkl')
    print(f"✓ Saved feature scaler")
    
    # Save metadata
    metadata = {
        'data_source': 'ICRISAT-District Level Data.csv',
        'total_samples': len(df),
        'features': feature_names,
        'crop_classes': {v: k for k, v in crop_encoding.items()},
        'models_trained': ['Decision Tree', 'K-Means Clustering', 'Linear Regression']
    }
    
    metadata_file = Path(save_dir) / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved model metadata")
    
    # ============================================
    # Summary Report
    # ============================================
    print("\n" + "=" * 70)
    print("✅ MODEL RETRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 SUMMARY:")
    print(f"  Decision Tree Accuracy: {results['decision_tree']['accuracy']:.2%}")
    print(f"  K-Means Silhouette Score: {results['kmeans']['silhouette_score']:.4f}")
    print(f"  Linear Regression R²: {results['linear_regression']['r2']:.4f}")
    print(f"\n📁 Models saved in: {save_dir}/")
    print(f"  • decision_tree_classifier.pkl")
    print(f"  • knn_clustering.pkl")
    print(f"  • linear_regression.pkl")
    print(f"  • scaler.pkl")
    print(f"  • model_results.json")
    print(f"  • model_metadata.json")
    print("\n" + "=" * 70)
    
    return results, data


if __name__ == "__main__":
    csv_path = "d:/AIOEL/ICRISAT-District Level Data.csv"
    save_dir = "models"
    
    results, data = retrain_all_models(csv_path, save_dir)
    print("\n🎉 Ready to update GUI with new models!")
