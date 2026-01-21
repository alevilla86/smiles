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

### Running Tests
```bash
python leishmania_activity_tests.py  # Tests with USAL compounds
python tests.py                       # General tests
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Entry Point
- `main.py` - Streamlit web application for interactive predictions

### ML Models (Two Approaches)

**Random Forest (Active/Primary):**
- `lesihmania_activity_predictions.py` - Prediction module (note: filename has typo)
- `leishmania_donovani_activity_trainer.py` - Training pipeline
- Model: `models/leishmania_donovani_model_v4.pkl`
- Uses 500-tree Random Forest with Morgan fingerprints (radius=2, 2048 bits)

**Semi-Supervised VAE (Experimental):**
- `leishmania_semi_supervised.py` - PyTorch VAE + classifier
- Models: `models/leishmania_donovani_vae_ae.pth`, `models/leishmania_donovani_vae_clf.pth`
- Architecture: 2048 → 512 → 256 (latent) → classifier

### Core Modules
- `constants.py` - Configuration, model paths, hardcoded benzimidazole compounds
- `compound_properties.py` - Molecular descriptor calculation (LogP, TPSA, H-bond donors/acceptors)
- `similarity_calc.py` - Tanimoto and Cosine similarity calculations
- `compound_parser.py` - PubChem JSON parsing

### Data Files
- `l_donovani_ACTIVE.txt` / `l_donovani_NOT_ACTIVE.txt` - Training compound SMILES lists

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
