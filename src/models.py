"""
Machine learning models module: Decision Tree, KNN Clustering, and Linear Regression.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from utils import save_model, load_model

class AgriculturalModels:
    """Container for all three models and their training/evaluation logic."""
    
    def __init__(self):
        self.dt_classifier = None
        self.knn_model = None
        self.lr_model = None
        
        self.dt_results = {}
        self.knn_results = {}
        self.lr_results = {}
    
    # ==================== DECISION TREE CLASSIFIER ====================
    
    def train_decision_tree(self, X_train, y_train, X_test=None, y_test=None, max_depth=10):
        """Train Decision Tree Classifier for crop recommendation."""
        self.dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.dt_classifier.fit(X_train, y_train)
        
        if X_test is not None and y_test is not None:
            y_pred = self.dt_classifier.predict(X_test)
            
            self.dt_results['accuracy'] = accuracy_score(y_test, y_pred)
            self.dt_results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            self.dt_results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            self.dt_results['y_pred'] = y_pred
            self.dt_results['y_test'] = y_test
            
            print(f"Decision Tree Classifier Results:")
            print(f"  Accuracy:  {self.dt_results['accuracy']:.4f}")
            print(f"  Precision: {self.dt_results['precision']:.4f}")
            print(f"  Recall:    {self.dt_results['recall']:.4f}")
        
        # Feature importance
        self.dt_results['feature_importance'] = self.dt_classifier.feature_importances_
        
        return self.dt_classifier
    
    def get_feature_importance(self, feature_names=None):
        """Get and rank feature importance from Decision Tree."""
        if self.dt_classifier is None:
            raise ValueError("Decision Tree not trained yet")
        
        importances = self.dt_classifier.feature_importances_
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return importances
    
    # ==================== KNN CLUSTERING ====================
    
    def train_knn_clustering(self, X, n_clusters=3):
        """Train KNN Clustering for soil profile segmentation."""
        self.knn_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.knn_model.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, clusters)
        self.knn_results['silhouette_score'] = silhouette_avg
        self.knn_results['clusters'] = clusters
        self.knn_results['n_clusters'] = n_clusters
        
        print(f"KNN Clustering Results:")
        print(f"  Number of Clusters: {n_clusters}")
        print(f"  Silhouette Score:   {silhouette_avg:.4f}")
        
        return self.knn_model, clusters
    
    def predict_cluster(self, X):
        """Predict cluster for new data points."""
        if self.knn_model is None:
            raise ValueError("KNN model not trained yet")
        return self.knn_model.predict(X)
    
    # ==================== LINEAR REGRESSION ====================
    
    def train_linear_regression(self, X_train, y_train, X_test=None, y_test=None):
        """Train Linear Regression for crop yield prediction."""
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)
        
        train_pred = self.lr_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        self.lr_results['train_rmse'] = train_rmse
        self.lr_results['train_mae'] = train_mae
        self.lr_results['train_r2'] = train_r2
        
        if X_test is not None and y_test is not None:
            test_pred = self.lr_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            self.lr_results['test_rmse'] = test_rmse
            self.lr_results['test_mae'] = test_mae
            self.lr_results['test_r2'] = test_r2
            self.lr_results['y_test'] = y_test
            self.lr_results['y_pred'] = test_pred
            
            # Calculate residuals
            residuals = y_test - test_pred
            self.lr_results['residuals'] = residuals
            
            print(f"Linear Regression Results (Test Set):")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  MAE:  {test_mae:.4f}")
            print(f"  R²:   {test_r2:.4f}")
        
        print(f"Linear Regression Results (Train Set):")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")
        
        return self.lr_model
    
    def predict_yield(self, X):
        """Predict crop yield for new data points."""
        if self.lr_model is None:
            raise ValueError("Linear Regression model not trained yet")
        return self.lr_model.predict(X)
    
    def get_regression_coefficients(self, feature_names=None):
        """Get regression coefficients."""
        if self.lr_model is None:
            raise ValueError("Linear Regression model not trained yet")
        
        coeffs = self.lr_model.coef_
        
        if feature_names is not None:
            coeff_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coeffs
            }).sort_values('coefficient', ascending=False, key=abs)
            return coeff_df
        else:
            return coeffs
    
    # ==================== MODEL PERSISTENCE ====================
    
    def save_all_models(self):
        """Save all trained models."""
        if self.dt_classifier is not None:
            save_model(self.dt_classifier, 'decision_tree_classifier')
        if self.knn_model is not None:
            save_model(self.knn_model, 'knn_clustering')
        if self.lr_model is not None:
            save_model(self.lr_model, 'linear_regression')
        print("All models saved successfully")
    
    def load_all_models(self):
        """Load all trained models."""
        try:
            self.dt_classifier = load_model('decision_tree_classifier')
            self.knn_model = load_model('knn_clustering')
            self.lr_model = load_model('linear_regression')
            print("All models loaded successfully")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
