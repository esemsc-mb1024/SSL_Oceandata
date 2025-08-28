import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# import your preprocessing utils
from sslearn.preprocessing.preprocessing import nearest_neighbor_fill, pad_to_200

"""
Sigma0 WV Postprocessing Script
===============================

This script postprocesses raw Sentinel-1 WV sigma⁰ `.npy` arrays to prepare them 
for training self-supervised learning (SSL) models.
"""


# ----------------------------
# Config
# ----------------------------
output_dir = Path("/Path/to/Sigma0_WV")   # directory with existing .npy files
processed_dir = output_dir / "processed"  # new folder for processed files
processed_dir.mkdir(parents=True, exist_ok=True)

def main():
    npy_files = list(output_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {output_dir}")
        return

    for npy_file in tqdm(npy_files, desc="Postprocessing"):
        try:
            # Step 1: load array
            arr = np.load(npy_file)

            # Step 2: fill NaNs
            arr = nearest_neighbor_fill(arr)

            # Step 3: pad if smaller
            arr = pad_to_200(arr)

            # Step 4: save into processed folder
            out_path = processed_dir / npy_file.name
            np.save(out_path, arr.astype(np.float16, copy=False))

        except Exception as e:
            print(f"Error processing {npy_file}: {e}")

    print(f"Processed arrays saved to {processed_dir}")

if __name__ == "__main__":
    main()