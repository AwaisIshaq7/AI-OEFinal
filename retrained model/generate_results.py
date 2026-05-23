"""
Generate model evaluation results from trained models
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, silhouette_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from prepare_icrisat_data import prepare_icrisat_data


def generate_model_results():
    """Generate and save model evaluation results"""
    
    print("📊 Generating model evaluation results...")
    
    # Load data
    csv_path = "d:/AIOEL/ICRISAT-District Level Data.csv"
    data, df = prepare_icrisat_data(csv_path)
    
    # Load models
    dt = joblib.load('models/decision_tree_classifier.pkl')
    kmeans = joblib.load('models/knn_clustering.pkl')
    lr = joblib.load('models/linear_regression.pkl')
    
    # Get data components
    X_class = data['X_class']
    y_class = data['y_class']
    crop_encoding = data['crop_encoding']
    X_cluster = data['X_cluster']
    X_reg = data['X_reg']
    y_reg = data['y_reg']
    
    # Evaluate models
    results = {}
    
    # Decision Tree
    y_pred_dt = dt.predict(X_class)
    dt_accuracy = accuracy_score(y_class, y_pred_dt)
    results['decision_tree'] = {
        'accuracy': float(dt_accuracy),
        'n_features': X_class.shape[1],
        'n_classes': len(np.unique(y_class)),
        'feature_importance': {str(i): float(imp) for i, imp in enumerate(dt.feature_importances_)},
        'crop_mapping': {str(v): k for k, v in crop_encoding.items()}
    }
    
    # K-Means
    clusters = kmeans.predict(X_cluster)
    km_silhouette = silhouette_score(X_cluster, clusters)
    results['kmeans'] = {
        'silhouette_score': float(km_silhouette),
        'n_clusters': 4,
        'inertia': float(kmeans.inertia_),
        'n_samples': X_cluster.shape[0]
    }
    
    # Linear Regression
    y_pred_lr = lr.predict(X_reg)
    lr_rmse = np.sqrt(mean_squared_error(y_reg, y_pred_lr))
    lr_mae = mean_absolute_error(y_reg, y_pred_lr)
    lr_r2 = r2_score(y_reg, y_pred_lr)
    
    results['linear_regression'] = {
        'rmse': float(lr_rmse),
        'mae': float(lr_mae),
        'r2': float(lr_r2),
        'test_r2': float(lr_r2),
        'n_samples': len(y_reg),
        'y_test': y_reg[:50].tolist(),  # First 50 for visualization
        'y_pred': y_pred_lr[:50].tolist()
    }
    
    # Save results
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to models/model_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
    print(f"K-Means Silhouette Score: {km_silhouette:.4f}")
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Linear Regression RMSE: {lr_rmse:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    generate_model_results()
