from pathlib import Path

SIMILARITY_RESULTS_DIR = Path("similarity_results")
RECOMMENDATIONS_DIR = Path("recommendations")

# Leishmania model configuration
LEISHMANIA_MODEL_PATH = "models/leishmania_model_v2.pkl"
LEISHMANIA_SPECIES = ["Leishmania major", "Leishmania donovani", "Leishmania infantum", 
                      "Leishmania mexicana", "Leishmania braziliensis"]
MAX_VALUE_UM_IC50 = 10.0