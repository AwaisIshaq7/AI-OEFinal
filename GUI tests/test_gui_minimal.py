#!/usr/bin/env python
"""Minimal test to isolate GUI startup issue"""

import sys
import os

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("[1] Starting minimal GUI test...", flush=True)

sys.path.insert(0, 'src')

import tkinter as tk
print("[2] Tkinter imported", flush=True)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

print("[3] Dependencies imported", flush=True)

def load_models():
    """Load all trained models with error handling"""
    try:
        print("[4] Starting load_models()", flush=True)
        models_dir = Path('models')
        
        print("[4a] Loading DT classifier...", flush=True)
        dt_classifier = joblib.load(models_dir / 'decision_tree_classifier.pkl')
        
        print("[4b] Loading KMeans model...", flush=True)
        kmeans_model = joblib.load(models_dir / 'knn_clustering.pkl')
        
        print("[4c] Loading LR model...", flush=True)
        lr_model = joblib.load(models_dir / 'linear_regression.pkl')
        
        print("[4d] Loading scaler...", flush=True)
        scaler_path = models_dir / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            scaler = StandardScaler()
            dummy_data = np.array([[40, 10, 30, 15, 50, 6.0, 500],
                                   [100, 50, 80, 35, 90, 8.0, 2000]])
            scaler.fit(dummy_data)
        
        print("[4e] Loading dataset...", flush=True)
        dataset = pd.read_csv('ICRISAT-District Level Data.csv')
        
        print("[4f] Loading model results...", flush=True)
        with open(models_dir / 'model_results.json', 'r') as f:
            model_results = json.load(f)
        
        print("[4g] Printing metrics...", flush=True)
        print(f"   DT Accuracy: {model_results['decision_tree']['accuracy']:.2%}")
        print(f"   KMeans Silhouette: {model_results['knn_clustering']['silhouette_score']:.4f}")
        print(f"   LR R-squared: {model_results['linear_regression']['r2']:.4f}")
        
        print("[5] load_models() completed successfully", flush=True)
        return True
        
    except Exception as e:
        print(f"[ERROR] load_models() failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

print("[6] Loading models...", flush=True)
if not load_models():
    print("[ERROR] Model loading failed!", flush=True)
    sys.exit(1)

print("[7] Creating Tkinter root window...", flush=True)
root = tk.Tk()
root.title("Test Window")
root.geometry("400x200")

print("[8] Adding test label...", flush=True)
label = tk.Label(root, text="GUI is running!")
label.pack(pady=20)

print("[9] Creating destroy button...", flush=True)
btn = tk.Button(root, text="Close", command=root.quit)
btn.pack()

print("[10] Starting mainloop...", flush=True)
print("[NOTE] Window should appear. Click Close to exit.", flush=True)

root.after(3000, root.quit)  # Auto-close after 3 seconds
root.mainloop()

print("[11] GUI test completed successfully!", flush=True)
