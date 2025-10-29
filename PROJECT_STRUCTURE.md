# Project Structure - Successfully Created! ✅

## Overview
The Order Reconstruction Challenge codebase has been successfully set up with a modular, config-driven architecture.

## Directory Structure Created

```
order-reconstruction/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # 🔧 CONFIGURATION FILE (edit this!)
│   ├── main.py                  # Main pipeline entry point
│   ├── base_algorithm.py        # Abstract base class for algorithms
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessor.py          # Signal preprocessing
│   ├── feature_extractor.py     # Feature engineering
│   ├── inference.py             # Order inference methods
│   ├── output_manager.py        # Results management
│   ├── utils.py                 # Helper functions
│   └── algorithms/              # Algorithm implementations
│       ├── __init__.py
│       └── signal_processing.py # Baseline algorithm
│
├── notebooks/                    # Jupyter notebooks
│   ├── README.md
│   └── 00_quick_start.ipynb     # Getting started guide
│
├── data/                         # Dataset directory
│   └── .gitignore
│
├── results/                      # Experiment outputs
│   └── README.md
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation

```

## Key Files and Their Purpose

### Core Pipeline Files

1. **`src/config.py`** - Central configuration
   - Set algorithm to use (`ALGORITHM_NAME`)
   - Configure preprocessing options
   - Set feature extraction parameters
   - Define experiment name and output settings

2. **`src/main.py`** - Main pipeline script
   - Orchestrates: Load → Preprocess → Extract → Infer → Save
   - Run with: `python -m src.main`

3. **`src/base_algorithm.py`** - Algorithm interface
   - Abstract base class that all algorithms must inherit from
   - Defines methods: `preprocess()`, `extract_features()`, `infer_order()`

### Data Handling

4. **`src/data_loader.py`** - Data loading
   - Reads CSV files from `data/` directory
   - Handles batch loading with optional limits
   - Validates data structure

5. **`src/preprocessor.py`** - Signal preprocessing
   - Normalization
   - Filtering (lowpass, highpass, bandpass)
   - Detrending
   - Outlier removal

### Feature Engineering

6. **`src/feature_extractor.py`** - Feature extraction
   - Time-domain features (RMS, kurtosis, peak, energy, etc.)
   - Frequency-domain features (FFT, spectral features)
   - Tachometer features (RPM statistics)
   - Bearing fault-band power

### Inference & Output

7. **`src/inference.py`** - Order reconstruction
   - Monotonic ranking
   - PCA-based ordering
   - Weighted ranking
   - Spearman footrule evaluation

8. **`src/output_manager.py`** - Results management
   - Saves predictions in submission format
   - Exports features to CSV
   - Generates plots
   - Creates summary reports

9. **`src/utils.py`** - Utilities
   - Visualization helpers
   - Normalization functions
   - Bearing fault frequency calculations
   - Order summary printing

### Algorithm Implementation

10. **`src/algorithms/signal_processing.py`** - Baseline algorithm
    - Extracts signal processing features
    - Uses monotonic ranking for ordering
    - Serves as template for new algorithms

## How to Use

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place data in `data/` directory**
   - Download dataset from challenge page
   - Extract CSV files to `data/`

3. **Configure experiment** (`src/config.py`):
   ```python
   ALGORITHM_NAME = "signal_processing"
   OUTPUT_CONFIG = {
       "experiment_name": "experiment_01",  # Change for each run
       ...
   }
   ```

4. **Run pipeline:**
   ```bash
   python -m src.main
   ```

5. **Check results:**
   - Results saved to `results/experiment_01/`
   - Contains: submission.csv, features.csv, plots, report

### Using Notebooks

Open the quick start guide:
```bash
jupyter notebook notebooks/00_quick_start.ipynb
```

This notebook demonstrates:
- Loading data
- Extracting features
- Using algorithms
- Visualizing results

## Creating New Algorithms

To implement your own approach:

1. **Create new algorithm file:**
   ```python
   # src/algorithms/my_algorithm.py
   from src.base_algorithm import BaseAlgorithm
   
   class MyAlgorithm(BaseAlgorithm):
       def preprocess(self, data):
           # Your preprocessing
           return data
       
       def extract_features(self, data):
           # Your features
           return features_df
       
       def infer_order(self, features):
           # Your ordering logic
           return predicted_order
   ```

2. **Register in config:**
   ```python
   # src/config.py
   from src.algorithms.my_algorithm import MyAlgorithm
   
   ALGORITHM_NAME = "my_algorithm"
   ```

3. **Run and iterate!**

## Experiment Workflow

```
1. Explore data (notebooks)
   ↓
2. Develop algorithm (src/algorithms/)
   ↓
3. Configure (src/config.py)
   ↓
4. Run pipeline (python -m src.main)
   ↓
5. Analyze results (results/experiment_name/)
   ↓
6. Iterate (adjust config, try new features)
```

## Key Configuration Options

### Algorithm Selection
- `ALGORITHM_NAME`: Which algorithm to use
- `ALGORITHM_CONFIG`: Algorithm-specific parameters

### Preprocessing
- `normalize`: Z-score normalization
- `detrend`: Remove linear trend
- `filter_type`: Apply frequency filters
- `remove_outliers`: Clip extreme values

### Features
- Time-domain: RMS, kurtosis, peak, energy
- Frequency-domain: FFT, spectral features
- Fault-bands: Power at bearing frequencies
- Tachometer: RPM statistics

### Output
- `save_features`: Export feature matrix
- `save_plots`: Generate visualizations
- `experiment_name`: Unique name for results

## Dependencies

Core requirements:
- numpy, pandas: Data manipulation
- scipy: Signal processing
- scikit-learn: ML utilities
- matplotlib: Visualization

## Next Steps

1. ✅ Download dataset → place in `data/`
2. ✅ Review `notebooks/00_quick_start.ipynb`
3. ✅ Run baseline: `python -m src.main`
4. ✅ Experiment with features in `src/config.py`
5. ✅ Develop custom algorithms in `src/algorithms/`
6. ✅ Submit best result to leaderboard!

## Challenge Details

- **Metric**: Spearman footrule distance (lower is better)
- **Prize**: $26,000 total ($12k first place)
- **Goal**: Reconstruct chronological order of scrambled bearing degradation recordings

Good luck with the competition! 🚀
