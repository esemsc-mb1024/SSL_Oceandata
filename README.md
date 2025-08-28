# Self-Supervised Learning on Sentinel-1 WV Data

This repository contains code for training and evaluating self-supervised learning (SSL) models (SimCLR and DINO) on Sentinel-1 Wave Mode (WV) Synthetic Aperture Radar (SAR) imagery.

---

## Data Download

### Unlabelled WV Data
The training data used in this project comes from **Sentinel-1 WV SLC products**, distributed by the [Alaska Satellite Facility (ASF) DAAC](https://asf.alaska.edu/).

To access the data:

1. **Create an ASF Account**  
   Sign up for a free account here: [https://urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov).

2. **Set up your credentials**  
   Store your login in a `.netrc` file in your home directory:
   ```bash
   machine urs.earthdata.nasa.gov
   login YOURUSERNAME
   password YOURPASSWORD

3. **Install ASF Python client**
   pip install asf_search

   **Example Query**
   import asf_search as asf

    results = asf.search(
        platform="Sentinel-1",
        processingLevel="SLC",
        beamMode="WV",
        start="2019-01-01T00:00:00Z",
        end="2019-01-31T23:59:59Z",
        intersectsWith="POLYGON((-74.7 33.7, -72.6 29.6, -59.5 31.0, -57.4 42.2, -65.9 42.7, -74.7 33.7))"
    )
    
    print(f"Found {len(results)} scenes")

    Note: An internal Python script (not included here) was provided to download WV data. The process is relatively slow and memory intensive — around 50–100 images per hour.


### Labelled Dataset
The manually annotated dataset used for transfer learning is TenGeoP-SARwv, available at:
https://www.seanoe.org/data/00456/56796/
Contains 37,000+ images across 10 geophysical classes.
For this project, a subset of 1,000 images was used for transfer learning.

Model Training
Extract raw sigma⁰ backscatter images from the downloaded data:
python src/sslearn/preprocessing/sigma0.py
(Update the paths inside the script to point to your local data.)

Preprocess the data for training:
python src/sslearn/preprocessing/preprocessing_run.py

Train SSL models:
python src/sslearn/training/training_simclr.py
python src/sslearn/training/training_dino.py


Transfer Learning
Preprocess the labelled dataset:
python src/sslearn/preprocessing/preprocessed_tiff.py

Run transfer learning:
python src/sslearn/evaluation/transfer_dino.py
python src/sslearn/evaluation/transfer_simclr.py

Baseline comparisons:
python src/sslearn/evaluation/pretrained_tl.py
python src/sslearn/classical/baseline_lr.py
(Ensure models have been trained first, as no pretrained weights are included in this repository.)

Clustering
To run k-means clustering analysis for the SimCLR model:
python src/sslearn/clustering/cluster_run.py
(Ensure models have been trained first, as no pretrained weights are included in this repository.)






