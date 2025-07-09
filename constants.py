from pathlib import Path

SIMILARITY_RESULTS_DIR = Path("similarity_results")
RECOMMENDATIONS_DIR = Path("recommendations")

# Leishmania model configuration
LEISHMANIA_MODEL_PATH = "models/leishmania_model_v3.pkl"
LEISHMANIA_DONOVANI_MODEL_PATH = "models/leishmania_donovani_model_v3.pkl"

LEISHMANIA_SPECIES = ["Leishmania major", "Leishmania donovani", "Leishmania infantum", 
                      "Leishmania mexicana", "Leishmania braziliensis"]
LEISHMANIA_SPECIES_DONOVANI = ["Leishmania donovani"]
LEISHMANIA_SPECIES_NOT_DONOVANI = ["Leishmania major", "Leishmania infantum", 
                      "Leishmania mexicana", "Leishmania braziliensis"]

MAX_VALUE_UM_IC50 = 10.0