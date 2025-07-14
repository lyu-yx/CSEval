#!/usr/bin/env python3
"""
Script to organize projected scores from Ranking_whole/gt_continuous.csv
into individual ranking files matching the existing Ranking folder structure.
"""

import os
import pandas as pd
from pathlib import Path

def extract_filename_from_path(file_path):
    """Extract filename without extension from full path"""
    return Path(file_path).stem

def organize_projected_scores():
    """Organize projected scores into TrainDataset and TestDataset Ranking_proj folders"""
    
    # Read the projected scores CSV
    csv_path = "Ranking_whole/gt_continuous.csv"
    print(f"Reading projected scores from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} projected scores")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found!")
        return
    
    # Create output directories
    train_ranking_dir = Path("dataset/TrainDataset/Ranking_proj")
    test_ranking_dir = Path("dataset/TestDataset/COD10K/Ranking_proj")
    
    train_ranking_dir.mkdir(parents=True, exist_ok=True)
    test_ranking_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directories:")
    print(f"  {train_ranking_dir}")
    print(f"  {test_ranking_dir}")
    
    # Counters for tracking
    train_count = 0
    test_count = 0
    
    # Process each row in the CSV
    for idx, row in df.iterrows():
        img_path = row['img_path']
        score = row['score']
        
        # Extract filename from image path
        filename = extract_filename_from_path(img_path)
        
        # Determine if it's training or testing data based on path
        if "/TrainDataset/" in img_path:
            # Save to training directory
            output_file = train_ranking_dir / f"{filename}.txt"
            train_count += 1
        elif "/TestDataset/" in img_path:
            # Save to testing directory  
            output_file = test_ranking_dir / f"{filename}.txt"
            test_count += 1
        else:
            print(f"Warning: Unknown path structure for {img_path}")
            continue
        
        # Round score to match existing format (integer values)
        rounded_score = score
        
        # Write score to individual text file
        with open(output_file, 'w') as f:
            f.write(str(rounded_score))
        
        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1}/{len(df)} entries...")
    
    print(f"\n Successfully organized projected scores:")
    print(f"   Training files: {train_count}  {train_ranking_dir}")
    print(f"   Testing files:  {test_count}  {test_ranking_dir}")
    print(f"   Total processed: {train_count + test_count}")
    
    # Verify some files were created
    if train_count > 0:
        sample_train_files = list(train_ranking_dir.glob("*.txt"))[:3]
        print(f"\n Sample training files created:")
        for f in sample_train_files:
            with open(f, 'r') as file:
                content = file.read().strip()
            print(f"   {f.name}: {content}")
    
    if test_count > 0:
        sample_test_files = list(test_ranking_dir.glob("*.txt"))[:3]
        print(f"\n Sample testing files created:")
        for f in sample_test_files:
            with open(f, 'r') as file:
                content = file.read().strip()
            print(f"   {f.name}: {content}")

def verify_organization():
    """Verify the organization was successful"""
    print("\n Verification:")
    
    # Check directories exist
    train_dir = Path("dataset/TrainDataset/Ranking_proj")
    test_dir = Path("dataset/TestDataset/COD10K/Ranking_proj")
    
    if train_dir.exists():
        train_files = list(train_dir.glob("*.txt"))
        print(f"    Training Ranking_proj: {len(train_files)} files")
    else:
        print(f"    Training Ranking_proj directory not found")
    
    if test_dir.exists():
        test_files = list(test_dir.glob("*.txt"))
        print(f"    Testing Ranking_proj: {len(test_files)} files")
    else:
        print(f"    Testing Ranking_proj directory not found")
    
    # Compare with original CSV
    try:
        df = pd.read_csv("Ranking_whole/gt_continuous.csv")
        total_csv_entries = len(df)
        total_created_files = len(train_files) + len(test_files) if train_dir.exists() and test_dir.exists() else 0
        print(f"   CSV entries: {total_csv_entries}, Created files: {total_created_files}")
        
        if total_csv_entries == total_created_files:
            print(f"   Perfect match! All entries organized successfully.")
        else:
            print(f"     Mismatch detected. Please check for errors.")
    except:
        print(f"     Could not verify against original CSV")

if __name__ == "__main__":
    print("Organizing Projected Scores into Ranking_proj Folders")
    print("=" * 60)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Organize the scores
    organize_projected_scores()
    
    # Verify the results
    verify_organization()
    
    print("\n Organization complete!")