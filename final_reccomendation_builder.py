import pandas as pd
from pathlib import Path

def combine_excel_files():
    print("Recomendaciones finales combinadas y guardadas en 'RECOMENDACION_FINAL.xlsx'")
    folder_path = Path("recommendations")
    excel_files = list(folder_path.glob("*.csv"))
    dfs = []

    for file in excel_files:
        ref_smiles = file.stem.split("_")[0]
        df = pd.read_csv(file)
        df['ref_smiles'] = ref_smiles
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df = combined_df.sort_values(
        by=["probabilidad_leishmania", "tanimoto_similarity", "cosine_similarity"],
        ascending=[False, False, False]
    )

    combined_df.to_excel(folder_path / "RECOMENDACION_FINAL.xlsx", index=False)

def main(): # Renombra __main__ a main o un nombre más descriptivo
    combine_excel_files()

if __name__ == "__main__":
    main()