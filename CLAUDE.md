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

# Train XGBoost model
python leishmania_donovani_xgboost_trainer.py

# Train semi-supervised VAE model (experimental)
python leishmania_semi_supervised.py
```

### Analyzing Model Performance
```bash
# Analyze Random Forest accuracy vs n_estimators
python model_estimator_analysis.py --model rf

# Analyze XGBoost accuracy vs n_estimators
python model_estimator_analysis.py --model xgb

# Compare both models side-by-side
python model_estimator_analysis.py --model both
```

Uses 5-fold cross-validation to evaluate accuracy across a range of estimator values (10-1500). Generates plots saved to `models/` showing where accuracy plateaus and optimal n_estimators values.

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

**XGBoost (Gradient Boosting):**
- `leishmania_donovani_xgboost_trainer.py` - Training pipeline
- Uses gradient boosting with optimized hyperparameters

**Semi-Supervised VAE (Experimental):**
- `leishmania_semi_supervised.py` - PyTorch VAE + classifier training
- Models: `models/leishmania_donovani_vae_ae.pth`, `models/leishmania_donovani_vae_clf.pth`
- Architecture: 2048 → 512 → 256 (latent) → classifier

**Model Analysis:**
- `model_estimator_analysis.py` - Hyperparameter optimization tool
- Evaluates accuracy vs n_estimators using 5-fold cross-validation
- Identifies optimal and recommended estimator counts for both RF and XGBoost

### Core Modules
- `constants.py` - Configuration hub: model paths, UI settings (`UIConfig`), prediction thresholds, benzimidazole compounds
- `model_utils.py` - Model saving utilities with timestamped filenames and accuracy (`save_sklearn_model`, `save_pytorch_model`)

### Shared Utilities
- `fingerprint_utils.py` - Molecular fingerprint processing:
  - `get_fingerprint(smiles, radius=2, n_bits=2048)` - Convert SMILES to Morgan fingerprint
  - `smiles_to_fingerprints(smiles_list, label)` - Batch conversion with labels
  - `prepare_fingerprints(active_smiles, inactive_smiles)` - Prepare ML-ready feature matrices
  - Suppresses RDKit warnings globally on import
- `data_loader.py` - Training data I/O utilities:
  - `load_training_data()` - Load active/inactive SMILES from files
  - `save_training_data()` - Save SMILES to training files
  - `load_smiles_from_file(filepath)` - Load SMILES from a single file
  - `save_smiles_to_file(smiles_list, filepath)` - Save SMILES to a single file
  - File constants: `ACTIVE_SMILES_FILE`, `INACTIVE_SMILES_FILE`

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

## Agent Guidelines

### Code Quality (REQUIRED)
**Before implementing any code changes**, always use the `code-evaluator` agent to:
1. Review the proposed changes for potential issues
2. Identify any code duplication that would be introduced
3. Suggest improvements to the implementation approach

**After implementing code changes**, always use the `code-evaluator` agent to:
1. Verify the changes follow project patterns
2. Check for any new code duplication
3. Identify opportunities for refactoring

### Documentation Updates (REQUIRED)
After making code changes that affect functionality, always use the `docs-updater` agent to update README.md and CLAUDE.md as needed.
