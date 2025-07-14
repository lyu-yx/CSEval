#!/usr/bin/env python3
"""
Script to list all filenames from two directories and save as separate text files.
"""

from pathlib import Path

def save_filename_lists(folder1, folder2, output1="folder1_files.txt", output2="folder2_files.txt"):
    """
    Save lists of filenames from two directories
    
    Args:
        folder1: Path to first directory
        folder2: Path to second directory
        output1: Output filename for first directory's list
        output2: Output filename for second directory's list
    """
    # Get all filenames (without extension) from both directories
    files1 = sorted([f.stem for f in Path(folder1).glob("*")]) if Path(folder1).exists() else []
    files2 = sorted([f.stem for f in Path(folder2).glob("*")]) if Path(folder2).exists() else []
    
    # Save first directory's list
    with open(output1, 'w') as f:
        f.write("\n".join(files1))
    print(f"Saved {len(files1)} filenames from {folder1} to {output1}")
    
    # Save second directory's list
    with open(output2, 'w') as f:
        f.write("\n".join(files2))
    print(f"Saved {len(files2)} filenames from {folder2} to {output2}")
    
    return len(files1), len(files2)

if __name__ == "__main__":
    # Set your directory paths here
    FOLDER1 = "dataset/TestDataset/COD10K/Ranking"
    FOLDER2 = "dataset/TestDataset/COD10K/Ranking_proj"
    
    # Set output filenames
    OUTPUT1 = "ranking_files_list.txt"
    OUTPUT2 = "ranking_proj_files_list.txt"
    
    # Run the function
    count1, count2 = save_filename_lists(FOLDER1, FOLDER2, OUTPUT1, OUTPUT2)
    
    print(f"\nTotal files listed: {count1} (train) + {count2} (test) = {count1 + count2}")