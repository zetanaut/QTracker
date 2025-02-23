# QTracker: Muon Track Reconstruction and Momentum Training

## Overview
QTracker is a framework for reconstructing and analyzing muon tracks in particle physics experiments. This repository provides scripts for processing raw data, training models for track finding and momentum reconstruction, and evaluating reconstructed tracks using a quality metric.

## Prerequisites
Ensure that your dataset can be reliably organized into single-track data with identified mu+ and mu- events.

## Data Preparation
1. **Splitting Data into Mu+ and Mu- Tracks**
   
   Use `separate.py` to split J/psi, Drell-Yan (DY), or two-muon tracks (mu+/mu-) into separate files:
   ```sh
   python3 separate.py JPsi_Dump.root
   ```
   This will generate:
   - `JPsi_Dump_track1.root` (mu+ tracks)
   - `JPsi_Dump_track2.root` (mu- tracks)

2. **Generating Hit Arrays for Training**
   
   The `gen_training.py` script processes the separated muon tracks and prepares the necessary hit arrays for model training:
   ```sh
   python3 gen_training.py JPsi_Dump_track1.root JPsi_Dump_track2.root
   ```
   This will produce the following training data files:
   - `finder_training.root` (for track finding training)
   - `momentum_training-1.root` (for mu+ momentum training)
   - `momentum_training-2.root` (for mu- momentum training)

## Model Training

### 1. Training the Track Finder
```sh
python3 TrackFinder_train.py finder_training.root
```

### 2. Training the Momentum Reconstruction Models
```sh
python3 Momentum_training.py --output mom_mup.h5 momentum_training-1.root
python3 Momentum_training.py --output mom_mum.h5 momentum_training-2.root
```

Store the resulting models in the `QTracker_basic/models` directory.

## Testing the Tracker
To test the trained models on a dataset:
```sh
python3 QTracker_test.py JPsi_Dump.root
```
This will generate:
- `qtracker_reco.root` (Reconstructed output file)

## Evaluating Reconstruction Quality
### 1. Checking the Invariant Mass Spectrum
```sh
python3 imass_plot.py qtracker_reco.root
```
This script will plot the mass spectrum of your reconstructed events.

### 2. Training the Quality Metric Model (Chi-Squared Method)
```sh
python3 Qmetric_training.py qtracker_reco.root
```

## Notes
- Ensure that your dataset follows the expected RUS format before processing.
- The trained models should be stored in the correct directory (`QTracker_basic/models`) for proper operation.
- The scripts assume that dependencies such as ROOT, Python, and required ML libraries are properly installed.



