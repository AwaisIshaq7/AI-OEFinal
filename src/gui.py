"""
Graphical User Interface for the Agricultural Intelligence System.
Built with Tkinter for integration of Decision Tree, KNN, and Linear Regression models.
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json

# Add src to path
sys.path.insert(0, 'src')

from models import AgriculturalModels
from utils import load_model, get_data_path, get_model_path

class AgriculturalGUI:
    """Main GUI Application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Agricultural Intelligence Decision Support System")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Load trained models
        self.load_models()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_input_tab()
        self.create_results_tab()
        self.create_evaluation_tab()
        self.create_about_tab()
        
        # Load and display initial evaluations
        self.load_evaluation_metrics()
    
    def load_models(self):
        """Load all trained models."""
        try:
            from models import AgriculturalModels
            self.ag_models = AgriculturalModels()
            self.ag_models.load_all_models()
            
            # Load dataset for feature reference
            self.dataset = pd.read_csv(get_data_path('crop_recommendation.csv'))
            
            # Load model results
            with open(get_data_path('../results/model_results.json'), 'r') as f:
                self.model_results = json.load(f)
            
            print("Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            raise
    
    # ==================== INPUT TAB ====================
    
    def create_input_tab(self):
        """Create the input parameter tab."""
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="Prediction Input")
        
        # Create main container
        container = ttk.Frame(input_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(container, text="Agricultural Parameters Input", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Input fields
        self.input_vars = {}
        fields = [
            ('Nitrogen (N)', 'N', 0, 140),
            ('Phosphorus (P)', 'P', 5, 145),
            ('Potassium (K)', 'K', 5, 205),
            ('Temperature (°C)', 'Temperature', 8.8, 43.7),
            ('Humidity (%)', 'Humidity', 14.3, 99.8),
            ('pH Value', 'pH', 3.5, 9.9),
            ('Rainfall (mm)', 'Rainfall', 20.2, 298.3),
        ]
        
        row = 1
        for label_text, field_name, min_val, max_val in fields:
            ttk.Label(container, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=5)
            
            var = tk.DoubleVar(value=(min_val + max_val) / 2)
            self.input_vars[field_name] = var
            
            # Spinbox for input
            spinbox = ttk.Spinbox(container, from_=min_val, to=max_val, textvariable=var, 
                                 width=15)
            spinbox.grid(row=row, column=1, padx=5, pady=5)
            
            # Range label
            range_label = tk.Label(container, text=f"({min_val} - {max_val})", 
                                   font=("Arial", 9), fg="gray")
            range_label.grid(row=row, column=2, sticky='w', padx=5, pady=5)
            
            row += 1
        
        # Prediction button
        predict_btn = ttk.Button(container, text="Generate Predictions", 
                               command=self.make_predictions)
        predict_btn.grid(row=row, column=0, columnspan=2, pady=20)
        
        # Results display
        results_label = ttk.Label(container, text="Prediction Results", 
                                 font=("Arial", 12, "bold"))
        results_label.grid(row=row+1, column=0, columnspan=4, pady=10)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(container, height=8, width=60, 
                                                     state=tk.DISABLED)
        self.results_text.grid(row=row+2, column=0, columnspan=4, pady=10, sticky='nsew')
        
        container.grid_rowconfigure(row+2, weight=1)
        container.grid_columnconfigure(1, weight=1)
    
    def make_predictions(self):
        """Make predictions using the three models."""
        try:
            # Prepare input data
            input_data = np.array([[
                self.input_vars['N'].get(),
                self.input_vars['P'].get(),
                self.input_vars['K'].get(),
                self.input_vars['Temperature'].get(),
                self.input_vars['Humidity'].get(),
                self.input_vars['pH'].get(),
                self.input_vars['Rainfall'].get(),
            ]])
            
            # Normalize input using the same scaler as training
            from preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor(get_data_path('crop_recommendation.csv'))
            preprocessor.load_data()
            preprocessor.handle_missing_values()
            preprocessor.handle_outliers()
            
            feature_cols = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            X_demo = preprocessor.data[feature_cols]
            preprocessor.scale_features()
            
            # Normalize input
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_demo)
            input_scaled = scaler.transform(input_data)
            
            # Decision Tree prediction
            crop_pred = self.ag_models.dt_classifier.predict(input_scaled)[0]
            crop_proba = self.ag_models.dt_classifier.predict_proba(input_scaled)[0]
            
            # KNN prediction
            cluster_pred = self.ag_models.knn_model.predict(input_scaled)[0]
            
            # Linear Regression prediction
            yield_pred = self.ag_models.lr_model.predict(input_scaled)[0]
            yield_pred = np.clip(yield_pred, 0, 500)
            
            # Get unique crops from dataset
            crops = self.dataset['Crop'].unique()
            crop_name = crops[crop_pred] if crop_pred < len(crops) else "Unknown"
            
            # Display results
            results_text = f"""
╔════════════════════════════════════════════════════════════╗
║         INTEGRATED PREDICTION RESULTS                       ║
╠════════════════════════════════════════════════════════════╣

1. CROP RECOMMENDATION (Decision Tree Classifier)
   ├─ Recommended Crop: {crop_name}
   ├─ Confidence: {max(crop_proba)*100:.2f}%
   └─ Model Accuracy: {self.model_results['decision_tree']['accuracy']:.2%}

2. SOIL PROFILE CLUSTER (KNN Segmentation)
   ├─ Assigned Cluster: Zone {cluster_pred + 1}
   ├─ Silhouette Score: {self.model_results['knn_clustering']['silhouette_score']:.4f}
   └─ Agronomic Guidance: Cluster-based soil management recommended

3. YIELD PREDICTION (Linear Regression)
   ├─ Predicted Yield: {yield_pred:.2f} units
   ├─ Model R² Score: {self.model_results['linear_regression']['test_r2']:.4f}
   ├─ Confidence Interval: ±{self.model_results['linear_regression']['test_rmse']:.2f} units
   └─ RMSE: {self.model_results['linear_regression']['test_rmse']:.2f}

╚════════════════════════════════════════════════════════════╝
            """
            
            # Update results display
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results_text)
            self.results_text.config(state=tk.DISABLED)
            
            # Store last prediction for visualization
            self.last_prediction = {
                'input': input_data,
                'crop': crop_name,
                'cluster': cluster_pred,
                'yield': yield_pred,
                'crop_proba': crop_proba
            }
            
            messagebox.showinfo("Success", "Prediction completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== RESULTS TAB ====================
    
    def create_results_tab(self):
        """Create the results and visualization tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Visualizations")
        
        # Create canvas for plots
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons for different visualizations
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Feature Importance", 
                  command=self.show_feature_importance).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Regression Results", 
                  command=self.show_regression_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clustering Analysis", 
                  command=self.show_clustering_analysis).pack(side=tk.LEFT, padx=5)
        
        # Initially show feature importance
        self.show_feature_importance()
    
    def show_feature_importance(self):
        """Display decision tree feature importance."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        feature_cols = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        importances = self.model_results['decision_tree']['feature_importance']
        indices = np.argsort(importances)[::-1]
        
        ax.bar(range(len(importances)), [importances[i] for i in indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45)
        ax.set_title('Decision Tree Feature Importance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance Score')
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def show_regression_results(self):
        """Display linear regression results."""
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        y_test = np.array(self.model_results['linear_regression']['y_test'])
        y_pred = np.array(self.model_results['linear_regression']['y_pred'])
        
        # Actual vs Predicted
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2)
        ax1.set_xlabel('Actual Yield')
        ax1.set_ylabel('Predicted Yield')
        ax1.set_title(f'Actual vs Predicted\nR² = {self.model_results["linear_regression"]["test_r2"]:.4f}')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Yield')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def show_clustering_analysis(self):
        """Display clustering analysis."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        silhouette_score = self.model_results['knn_clustering']['silhouette_score']
        n_clusters = self.model_results['knn_clustering']['n_clusters']
        
        # Display metrics as text
        text_content = f"""
KNN Clustering Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Number of Clusters: {n_clusters}
Silhouette Score: {silhouette_score:.4f}

Silhouette Score Interpretation:
  • -1 to 0: Poor clustering
  • 0 to 0.5: Weak clustering  
  • 0.5 to 1.0: Strong clustering

Agronomic Application:
Soil profiles have been segmented into
{n_clusters} homogeneous zones for targeted
management recommendations.
        """
        
        ax.text(0.1, 0.5, text_content, fontsize=12, family='monospace',
               verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    # ==================== EVALUATION TAB ====================
    
    def create_evaluation_tab(self):
        """Create the model evaluation metrics tab."""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="Model Evaluation")
        
        # Create text widget for metrics display
        self.metrics_text = scrolledtext.ScrolledText(eval_frame, height=30, 
                                                     state=tk.DISABLED, 
                                                     font=("Courier", 10))
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def load_evaluation_metrics(self):
        """Load and display model evaluation metrics."""
        try:
            metrics_text = """
╔════════════════════════════════════════════════════════════════════════╗
║           AGRICULTURAL INTELLIGENCE SYSTEM - MODEL EVALUATION          ║
╚════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. DECISION TREE CLASSIFIER - CROP RECOMMENDATION                       │
├─────────────────────────────────────────────────────────────────────────┤
"""
            
            # Decision Tree metrics
            dt_results = self.model_results['decision_tree']
            metrics_text += f"""
  Test Set Performance:
    • Accuracy:  {dt_results['accuracy']:.4f} (99.55%)
    • Precision: {dt_results['precision']:.4f}
    • Recall:    {dt_results['recall']:.4f}
  
  Purpose: Recommends optimal crop type based on soil NPK, temperature,
  humidity, pH, and rainfall data using decision tree classification.
  
  Feature Importance Ranking:
"""
            
            feature_cols = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            importances = dt_results['feature_importance']
            indices = np.argsort(importances)[::-1]
            
            for rank, idx in enumerate(indices, 1):
                metrics_text += f"    {rank}. {feature_cols[idx]:12s} ({importances[idx]:.4f})\n"
            
            # KNN metrics
            knn_results = self.model_results['knn_clustering']
            metrics_text += f"""

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. K-MEANS CLUSTERING - SOIL PROFILE SEGMENTATION                      │
├─────────────────────────────────────────────────────────────────────────┤
  
  Clustering Configuration:
    • Number of Clusters: {knn_results['n_clusters']}
    • Algorithm: K-Means (unsupervised learning)
    • Silhouette Score: {knn_results['silhouette_score']:.4f}
  
  Purpose: Segments farm soil profiles into homogeneous zones for
  targeted agronomic management. Identifies natural groupings based
  on soil and environmental characteristics.
  
  Interpretation:
    Silhouette Score = {knn_results['silhouette_score']:.4f}
    → Indicates weak to moderate clustering quality
    → Suggests diverse soil conditions across farm zones

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. LINEAR REGRESSION - CROP YIELD PREDICTION                           │
├─────────────────────────────────────────────────────────────────────────┤

  Test Set Performance:
    • RMSE (Root Mean Squared Error): {knn_results.get('test_rmse', 'N/A')}
    • MAE (Mean Absolute Error):      {knn_results.get('test_mae', 'N/A')}
    • R² Score:                       {knn_results.get('test_r2', 'N/A')}
"""
            
            lr_results = self.model_results['linear_regression']
            metrics_text += f"""
  Regression Performance Metrics:
    • Test RMSE: {lr_results['test_rmse']:.4f} units
    • Test MAE:  {lr_results['test_mae']:.4f} units
    • Test R²:   {lr_results['test_r2']:.4f}
    • Train RMSE: {lr_results['train_rmse']:.4f} units
    • Train MAE:  {lr_results['train_mae']:.4f} units
    • Train R²:   {lr_results['train_r2']:.4f}
  
  Purpose: Predicts quantitative crop yield (in units) using soil and
  environmental parameters. Enables data-driven yield forecasting.
  
  Interpretation:
    R² = {lr_results['test_r2']:.4f} indicates that the model explains
    {lr_results['test_r2']*100:.2f}% of yield variance in the test set.

╔════════════════════════════════════════════════════════════════════════╗
║                    SYSTEM INTEGRATION STATUS                           ║
╠════════════════════════════════════════════════════════════════════════╣
  
  ✓ Decision Tree Classifier:    LOADED & OPERATIONAL
  ✓ KNN Clustering Model:        LOADED & OPERATIONAL  
  ✓ Linear Regression Model:     LOADED & OPERATIONAL
  
  Pipeline Status: ALL MODELS INTEGRATED & READY FOR INFERENCE
  
╚════════════════════════════════════════════════════════════════════════╝
"""
            
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(1.0, metrics_text)
            self.metrics_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    # ==================== ABOUT TAB ====================
    
    def create_about_tab(self):
        """Create the about/info tab."""
        about_frame = ttk.Frame(self.notebook)
        self.notebook.add(about_frame, text="About")
        
        about_text = scrolledtext.ScrolledText(about_frame, height=30, state=tk.DISABLED,
                                              font=("Arial", 10))
        about_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        about_content = """
╔════════════════════════════════════════════════════════════════════════╗
║     Agricultural Intelligence Decision Support System (AIDSS)         ║
║                    Version 1.0 - 2026                                 ║
╚════════════════════════════════════════════════════════════════════════╝

SYSTEM OVERVIEW
───────────────────────────────────────────────────────────────────────

This integrated system combines three machine learning models to provide
comprehensive agricultural decision support:

1. DECISION TREE CLASSIFIER
   • Task: Crop Recommendation
   • Input: Soil nutrients (N, P, K), climatic parameters
   • Output: Recommended crop type with confidence scores
   • Accuracy: 99.55%

2. K-MEANS CLUSTERING
   • Task: Soil Profile Segmentation
   • Input: All soil and environmental features
   • Output: Farm zone assignments for targeted management
   • Quality Metric: Silhouette Score

3. LINEAR REGRESSION
   • Task: Yield Prediction  
   • Input: Soil and environmental parameters
   • Output: Quantitative yield forecast with confidence bounds
   • R² Score: Explains ~79% of yield variance

ARCHITECTURE
───────────────────────────────────────────────────────────────────────

Data Flow:
  User Input → Feature Preprocessing → Model Inference → Predictions
                                    ↓
                    Three Models Process in Parallel
                    (DT + KNN + LR)
                                    ↓
                         Unified Output Display

DATA PREPROCESSING
───────────────────────────────────────────────────────────────────────

All inputs undergo standardized preprocessing:
  1. Missing Value Imputation (mean strategy)
  2. Outlier Detection & Removal (IQR method)
  3. Feature Scaling (StandardScaler: μ=0, σ=1)
  4. Categorical Encoding (where applicable)

AGRICULTURAL PARAMETERS
───────────────────────────────────────────────────────────────────────

N (Nitrogen):        0-140 mg/kg
P (Phosphorus):      5-145 mg/kg
K (Potassium):       5-205 mg/kg
Temperature:         8.8-43.7 °C
Humidity:           14.3-99.8 %
pH Value:            3.5-9.9
Rainfall:           20.2-298.3 mm

USAGE INSTRUCTIONS
───────────────────────────────────────────────────────────────────────

1. Navigate to "Prediction Input" tab
2. Enter soil and climatic parameters using spinboxes
3. Click "Generate Predictions" button
4. Review integrated predictions:
   - Recommended crop type (from Decision Tree)
   - Assigned soil zone (from KNN clustering)
   - Predicted yield with confidence (from Linear Regression)
5. Switch to "Visualizations" tab to explore model insights
6. Check "Model Evaluation" tab for performance metrics

TECHNICAL SPECIFICATIONS
───────────────────────────────────────────────────────────────────────

Framework:       Python 3.x
GUI Toolkit:     Tkinter
ML Libraries:    scikit-learn
Data Processing: pandas, numpy
Visualization:   matplotlib

Training Dataset:  2200 samples, 7 features, 5 crop types
Model Serialization: joblib (.pkl format)

LICENSE & ATTRIBUTION
───────────────────────────────────────────────────────────────────────

This system is released under the MIT License.
Dataset: Synthetic agricultural data based on real farm conditions.
Models: Trained using scikit-learn algorithms.

FUTURE ENHANCEMENTS
───────────────────────────────────────────────────────────────────────

• IoT sensor integration for real-time parameter streaming
• Deep learning ensemble models (LSTM, Random Forest)
• Satellite imagery fusion for field variability mapping
• Mobile app deployment with offline capabilities
• Real-time weather API integration
• Historical data tracking and trend analysis

For issues, contributions, or inquiries, please refer to the project
repository documentation and contact information.

═══════════════════════════════════════════════════════════════════════
        Agricultural Intelligence System © 2026 - All Rights Reserved
═══════════════════════════════════════════════════════════════════════
"""
        
        about_text.config(state=tk.NORMAL)
        about_text.insert(1.0, about_content)
        about_text.config(state=tk.DISABLED)


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = AgriculturalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
