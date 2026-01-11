#!/usr/bin/env python3
"""
Quick script to check if FER-2013 dataset is properly set up.
"""

import os
import sys

print("=" * 60)
print("🔍 Checking FER-2013 Dataset")
print("=" * 60)

data_dir = "data/fer2013"
required_files = ["train.csv", "test.csv"]
required_folders = ["train", "test"]

print(f"\nChecking directory: {data_dir}")

# Check if directory exists
if not os.path.exists(data_dir):
    print(f"❌ Directory not found: {data_dir}")
    print(f"\n📥 To fix:")
    print(f"   mkdir -p {data_dir}")
    print(f"   Then download dataset from: https://www.kaggle.com/datasets/msambare/fer2013")
    sys.exit(1)

print(f"✅ Directory exists: {data_dir}")

# Check for CSV files
print("\nChecking for CSV files...")
csv_found = False
for file in required_files:
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"   ✅ {file} ({size:,} bytes)")
        csv_found = True
    else:
        print(f"   ❌ {file} - NOT FOUND")

# Check for folder structure
print("\nChecking for folder structure...")
folders_found = False
for folder in required_folders:
    folder_path = os.path.join(data_dir, folder)
    if os.path.exists(folder_path):
        print(f"   ✅ {folder}/ - EXISTS")
        folders_found = True
    else:
        print(f"   ❌ {folder}/ - NOT FOUND")

# Summary
print("\n" + "=" * 60)
if csv_found:
    print("✅ Dataset ready! CSV format detected.")
    print("\nYou can now run:")
    print("  jupyter notebook notebooks/train_face_emotion.ipynb")
elif folders_found:
    print("✅ Dataset ready! Folder structure detected.")
    print("\nYou can now run:")
    print("  jupyter notebook notebooks/train_face_emotion.ipynb")
else:
    print("❌ Dataset NOT FOUND")
    print("\n📥 Download Instructions:")
    print("  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("  2. Click 'Download' (requires free Kaggle account)")
    print("  3. Extract zip file")
    print("  4. Copy train.csv and test.csv to: data/fer2013/")
    print("\nOr run: bash download_fer2013.sh")
    sys.exit(1)

print("=" * 60)

