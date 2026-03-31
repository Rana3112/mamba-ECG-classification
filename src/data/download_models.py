"""
Download all trained model files from Kaggle.
Run this in a Kaggle cell to create a zip with all .pth and .json files.
Then download the zip from /kaggle/working/models_export.zip
"""

import os
import zipfile

OUTPUT_DIR = "/kaggle/working/"
ZIP_PATH = os.path.join(OUTPUT_DIR, "models_export.zip")

# Collect all model files
files = []
for f in os.listdir(OUTPUT_DIR):
    if any(
        x in f.lower()
        for x in ["_summary.json", "_best.pth", "_latest.pth", "_results.json"]
    ):
        full = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(full):
            files.append(full)
            print(f"  Found: {f} ({os.path.getsize(full) / 1024 / 1024:.1f} MB)")

print(f"\nTotal: {len(files)} files")

# Create zip
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, os.path.basename(f))
        print(f"  Added: {os.path.basename(f)}")

print(f"\nZip created: {ZIP_PATH} ({os.path.getsize(ZIP_PATH) / 1024 / 1024:.1f} MB)")
print("Download it from the sidebar → Output tab")
