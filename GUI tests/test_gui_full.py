#!/usr/bin/env python
"""Test full GUI initialization step-by-step"""

import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("[1] Importing modules...", flush=True)
sys.path.insert(0, 'src')

import tkinter as tk
from gui import AgriculturalGUI

print("[2] Creating Tkinter root...", flush=True)
root = tk.Tk()

print("[3] Instantiating AgriculturalGUI...", flush=True)
try:
    app = AgriculturalGUI(root)
    print("[4] AgriculturalGUI created successfully", flush=True)
    
    print("[5] Auto-closing in 2 seconds...", flush=True)
    root.after(2000, root.quit)
    
    print("[6] Starting mainloop...", flush=True)
    root.mainloop()
    
    print("[7] GUI closed normally", flush=True)
    
except Exception as e:
    print(f"[ERROR] Exception occurred: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[8] Test completed!", flush=True)
