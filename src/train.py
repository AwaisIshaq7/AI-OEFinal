"""
Training pipeline - preprocesses data and trains all three models.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# Add src to path
sys.path.insert(0, 'src')

from preprocessing import DataPreprocessor
from models import AgriculturalModels
from utils import get_data_path, get_model_path
from download_data import download_crop_recommendation_data

def train_all_models():
    """Complete training pipeline."""
    
    print("=" * 60)
    print("AGRICULTURAL INTELLIGENCE SYSTEM - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Acquisition
    print("\n[STEP 1] Data Acquisition")
    print("-" * 60)
    download_crop_recommendation_data()
    
    # Step 2: Data Loading
    print("\n[STEP 2] Data Loading")
    print("-" * 60)
    preprocessor = DataPreprocessor(get_data_path('crop_recommendation.csv'))
    data = preprocessor.load_data()
    
    # Step 3: Data Preprocessing
    print("\n[STEP 3] Data Preprocessing")
    print("-" * 60)
    
    # Store original crop values before preprocessing
    y_crop = data['Crop'].copy()
    
    # Preprocess all features except Crop
    data_features = data.drop('Crop', axis=1)
    preprocessor.data = data_features
    
    # Manual preprocessing for features
    preprocessor.handle_missing_values(strategy='mean')
    preprocessor.handle_outliers()
    preprocessor.scale_features()
    
    # X contains preprocessed features
    X = preprocessor.data
    
    # Crop target (unencoded string values)
    # Linear Regression target: synthetic yield based on original features
    feature_cols = X.columns.tolist()
    
    # Create synthetic yield from original data (before scaling)
    original_data = pd.read_csv(get_data_path('crop_recommendation.csv'))
    y_yield = (original_data['N'] * 0.5 + original_data['P'] * 0.3 + original_data['K'] * 0.2 + 
               original_data['Temperature'] * 10 - original_data['Humidity'] * 5 + 
               original_data['pH'] * 15 + original_data['Rainfall'] * 2)
    y_yield = y_yield + np.random.normal(0, 50, len(y_yield))
    y_yield = np.clip(y_yield, 0, 500)  # Realistic yield range
    
    # Step 4: Model Training
    print("\n[STEP 4] Model Training")
    print("-" * 60)
    
    models = AgriculturalModels()
    
    # Split data for supervised models
    X_train, X_test, y_crop_train, y_crop_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42
    )
    
    _, _, y_yield_train, y_yield_test = train_test_split(
        X, y_yield, test_size=0.2, random_state=42
    )
    
    # Train Decision Tree Classifier
    print("\nTraining Decision Tree Classifier...")
    models.train_decision_tree(X_train, y_crop_train, X_test, y_crop_test, max_depth=10)
    
    # Train KNN Clustering
    print("\nTraining KNN Clustering...")
    models.train_knn_clustering(X, n_clusters=4)
    
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    models.train_linear_regression(X_train, y_yield_train, X_test, y_yield_test)
    
    # Step 5: Model Persistence
    print("\n[STEP 5] Model Serialization")
    print("-" * 60)
    models.save_all_models()
    
    # Step 6: Generate Results
    print("\n[STEP 6] Generating Results and Visualizations")
    print("-" * 60)
    
    results_summary = {
        'decision_tree': models.dt_results,
        'knn_clustering': models.knn_results,
        'linear_regression': models.lr_results,
    }
    
    # Save results to JSON
    results_path = get_data_path('../results/model_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays and non-serializable objects
        summary_clean = {}
        for model_name, results in results_summary.items():
            summary_clean[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    summary_clean[model_name][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    summary_clean[model_name][key] = float(value)
                elif hasattr(value, '__dict__'):
                    summary_clean[model_name][key] = str(value)
                else:
                    summary_clean[model_name][key] = value
        
        json.dump(summary_clean, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Create visualizations
    create_visualizations(models, feature_cols)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return models, X, y_crop, y_yield, feature_cols

def create_visualizations(models, feature_cols):
    """Create and save visualizations."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Feature Importance from Decision Tree
    ax1 = axes[0, 0]
    importances = models.dt_results['feature_importance']
    indices = np.argsort(importances)[::-1]
    ax1.bar(range(len(importances)), importances[indices])
    ax1.set_xticks(range(len(importances)))
    ax1.set_xticklabels([feature_cols[i] for i in indices], rotation=45)
    ax1.set_title('Decision Tree Feature Importance')
    ax1.set_ylabel('Importance')
    
    # Plot 2: Clustering Silhouette Score
    ax2 = axes[0, 1]
    silhouette = models.knn_results['silhouette_score']
    ax2.text(0.5, 0.5, f"Silhouette Score:\n{silhouette:.4f}", 
             ha='center', va='center', fontsize=16, transform=ax2.transAxes)
    ax2.set_title('KNN Clustering Quality Metric')
    ax2.axis('off')
    
    # Plot 3: Regression Predictions vs Actual
    ax3 = axes[1, 0]
    y_test = models.lr_results.get('y_test', [])
    y_pred = models.lr_results.get('y_pred', [])
    if len(y_test) > 0:
        ax3.scatter(y_test, y_pred, alpha=0.6)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax3.set_xlabel('Actual Yield')
        ax3.set_ylabel('Predicted Yield')
        ax3.set_title(f'Linear Regression: Actual vs Predicted\nR² = {models.lr_results.get("test_r2", 0):.4f}')
    
    # Plot 4: Residual Plot
    ax4 = axes[1, 1]
    residuals = models.lr_results.get('residuals', [])
    if len(residuals) > 0:
        ax4.scatter(y_pred, residuals, alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted Yield')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig(get_data_path('../results/model_visualizations.png'), dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {get_data_path('../results/model_visualizations.png')}")
    plt.close()

if __name__ == "__main__":
    try:
        train_all_models()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
