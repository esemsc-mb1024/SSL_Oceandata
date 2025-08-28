import os
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

"""
NetCDF to NumPy Conversion for Sentinel-1 Sigma0
================================================

This script extracts sigma⁰ backscatter arrays from Sentinel-1 WV NetCDF (.nc) 
files and saves them as `.npy` arrays for downstream preprocessing and training

"""

base_dir = "/Path/to/sentinel_data/"
output_dir = "/Path/to/Sigma0_WV/"
os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(base_dir):
    for file in tqdm(files):
        if file.endswith(".nc"):
            file_path = os.path.join(root, file)
            try:
                with Dataset(file_path, 'r') as nc_file:
                    sigma0 = nc_file.variables['sigma0'][0, :, :]
                    # masked array converted to Nan - Same as done in the netCDF file for the sigma0                             variable
                    if hasattr(sigma0, 'mask'):
                        sigma0 = sigma0.filled(np.nan)
                    np.save(os.path.join(output_dir, file.replace(".nc", ".npy")), sigma0)

            except Exception as e:

                print(f"Error processing {file_path}: {e}")