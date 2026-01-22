# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMILES is a machine learning application that predicts anti-leishmanial activity of chemical compounds using SMILES notation. Developed by Master's students from Universidad CENFOTEC in collaboration with Universidad de Salamanca's Department of Pharmaceutical Sciences.

The system uses Morgan fingerprints and Random Forest classification to predict whether chemical compounds may be active against Leishmania parasites (primarily L. donovani).

## Commands

### Running the Application
```bash
streamlit run main.py
```

### Training Models
```bash
# Train Random Forest model (fetches from ChEMBL + manual compounds)
python leishmania_donovani_activity_trainer.py

# Train semi-supervised VAE model (experimental)
python leishmania_semi_supervised.py
```

Models are saved with timestamped filenames including test accuracy:
- Format: `{model_name}_{YYYY-MM-DD_HH-MM-SS}_acc{accuracy}.{ext}`
- Example: `leishmania_donovani_rf_2025-01-22_14-30-45_acc0.9523.pkl`

### Running Tests
```bash
python leishmania_activity_tests.py  # Tests with USAL compounds
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Entry Point
- `main.py` - Streamlit web application for interactive predictions

### Prediction System
- `predictors.py` - Predictor factory with unified interface (`BasePredictor`)
  - `RandomForestPredictor` - Primary model using Morgan fingerprints
  - `VAEPredictor` - Experimental semi-supervised model
  - `create_predictor(PredictorType)` - Factory function to switch models
- To switch models, change `ACTIVE_PREDICTOR` in `main.py`

### ML Models

**Random Forest (Active/Primary):**
- `leishmania_donovani_activity_trainer.py` - Training pipeline
- Model: `models/leishmania_donovani_model_v4.pkl`
- Uses 500-tree Random Forest with Morgan fingerprints (radius=2, 2048 bits)

**Semi-Supervised VAE (Experimental):**
- `leishmania_semi_supervised.py` - PyTorch VAE + classifier training
- Models: `models/leishmania_donovani_vae_ae.pth`, `models/leishmania_donovani_vae_clf.pth`
- Architecture: 2048 → 512 → 256 (latent) → classifier

### Core Modules
- `constants.py` - Configuration hub: model paths, UI settings (`UIConfig`), prediction thresholds, benzimidazole compounds
- `model_utils.py` - Model saving utilities with timestamped filenames and accuracy (`save_sklearn_model`, `save_pytorch_model`)

### Data Files
- `training_data/l_donovani_ACTIVE.txt` / `l_donovani_NOT_ACTIVE.txt` - Training compound SMILES lists

## Key Dependencies
- **rdkit** - SMILES parsing, Morgan fingerprints, molecular properties
- **scikit-learn** - Random Forest classifier
- **torch** - Semi-supervised VAE model
- **chembl-webresource-client** - Fetching active compounds from ChEMBL database
- **streamlit** - Web UI
- **pandas/numpy** - Data processing

## Domain Context

- IC50 threshold: 10 µM (compounds with IC50 < 10 µM against L. donovani are considered active)
- Probability thresholds: Alta (≥0.8), Media (0.5-0.8), Baja (<0.5)
- Target species: L. donovani (primary), L. major, L. infantum, L. mexicana, L. braziliensis
- Training data sources: ChEMBL database + manually curated benzimidazole compounds
