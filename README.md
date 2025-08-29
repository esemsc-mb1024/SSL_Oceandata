  # SAR Wave Mode Project

  ## Data Sources

  ### Unlabelled WV Data
  The training data comes from **Sentinel-1 WV SLC products**, distributed by the [Alaska Satellite Facility (ASF) DAAC](https://asf.alaska.edu/).

  #### Access Instructions
  1. **Create an ASF Account**  
     Sign up for free: [https://urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)

  2. **Set up credentials**  
     Save your login details in a `.netrc` file in your home directory:

     ```bash
     machine urs.earthdata.nasa.gov
     login YOURUSERNAME
     password YOURPASSWORD
     ```

  3. **Install the ASF Python client**
     ```bash
     pip install asf_search
     ```

  4. **Example query**
     ```python
     import asf_search as asf

     results = asf.search(
         platform="Sentinel-1",
         processingLevel="SLC",
         beamMode="WV",
         start="2019-01-01T00:00:00Z",
         end="2019-01-31T23:59:59Z",
         intersectsWith="POLYGON((-74.7 33.7, -72.6 29.6, -59.5 31.0, -57.4 42.2, -65.9 42.7, -74.7 33.7))"
     )

    
     ```

  *Note*: An internal Python script (not included) was provided by ESTEC to download WV data. The process is slow and memory-intensive — expect ~50–100 images/hour.

  ---

  ### Labelled Dataset
  The manually annotated dataset used for transfer learning is **TenGeoP-SARwv**, available at:  
   [https://www.seanoe.org/data/00456/56796/](https://www.seanoe.org/data/00456/56796/)  

  - Contains **37,000+ images** across **10 geophysical classes**.  
  - For this project, a **subset of 1,000 images** was used for transfer learning.

  ---

  ## Model Training

  1. **Extract raw sigma⁰ backscatter images**
     ```bash
     python src/sslearn/preprocessing/sigma0.py
     ```
     *(Update the paths in the script to point to your local data.)*

  2. **Preprocess data for training**
     ```bash
     python src/sslearn/preprocessing/preprocessing_run.py
     ```

  3. **Train SSL models**
     ```bash
     python src/sslearn/training/training_simclr.py
     python src/sslearn/training/training_dino.py
     ```

  ---

  ##  Transfer Learning

  1. **Preprocess the labelled dataset**
     ```bash
     python src/sslearn/preprocessing/preprocessed_tiff.py
     ```

  2. **Run transfer learning**
     ```bash
     python src/sslearn/evaluation/transfer_dino.py
     python src/sslearn/evaluation/transfer_simclr.py
     ```

  3. **Baseline comparisons**
     ```bash
     python src/sslearn/evaluation/pretrained_tl.py
     python src/sslearn/classical/baseline_lr.py
     ```
     *(Ensure SSL models have been trained first; no pretrained weights are included.)*

  ---

  ## Clustering

  Run k-means clustering analysis for the SimCLR model:
  ```bash
  python src/sslearn/clustering/cluster_run.py
  ```
  *(Ensure SSL models have been trained first; no pretrained weights are included.)*





