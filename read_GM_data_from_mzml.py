from pyteomics import mzml
import os
from preprocess import SpectrumObject
import pandas as pd


path_folder = "/home/aguerrero@gaps_domain.ssr.upm.es/workspace/projects/K_pneumoniae/data/GM/mzml"

# walk through the folder and subfolders and find all .mzml files
mzml_files = []
for root, dirs, files in os.walk(path_folder):
    this_folder = [f"{root}/{file}" for file in files if file.endswith(".mzML")]
    mzml_files.extend(this_folder)


id_list = []
spectra = []
for mzml_file in mzml_files:
    print(f"Processing file {mzml_file}")
    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            spectrum = SpectrumObject(
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
            )
            spectrum = spectrum.preprocess_as_R()
            spectra.append(spectrum)
            id = mzml_file.split("/")[-1].split(".")[0].split("_")[2]
            id_list.append(id)


path_excel = "/home/aguerrero@gaps_domain.ssr.upm.es/workspace/projects/K_pneumoniae/data/GM/DB_conjunta.xlsx"
path_GM_AST = "/home/aguerrero@gaps_domain.ssr.upm.es/workspace/projects/K_pneumoniae/data/GM/GM_AST.xlsx"

df = pd.read_excel(path_excel)

df_AST = pd.read_excel(path_GM_AST)

# Match both dataset by Nº Micro on df_AST and Número de muestra in df
df_AST["Nº Micro"] = df_AST["Nº Micro"].astype(str)
df["Número de muestra"] = df["Número de muestra"].astype(str)

df_final = df.merge(
    df_AST, right_on="Nº Micro", left_on="Número de muestra", how="inner"
)

# Now, get from df_final the ones that match "Nº Espectro" with id_list
df_final["Nº Espectro"] = df_final["Nº Espectro"].astype(str)
df_final = df_final[df_final["Nº Espectro"].isin(id_list)]

# Which antibiotic do we want to check? In this case, CEFEPIME.
antibiotic = "CEFEPIME"
columns_to_get = ["Nº Espectro", antibiotic, antibiotic + ".1"]
df_final = df_final[columns_to_get]


# contruct the final X and Y datasets
X_int = []
X_mz = []
Y = []
for id in id_list:
    row = df_final[df_final["Nº Espectro"] == id]
    if row.empty:
        continue
    row = row.iloc[0]
    X_int.append(spectra[id_list.index(id)].intensity)
    X_mz.append(spectra[id_list.index(id)].mz)
    Y.append(row[antibiotic + ".1"])

# Save the data in a pkl
import pickle

gm_data = {
    "X_int": X_int,
    "X_mz": X_mz,
    "Y": Y,
}

with open("gm_data.pkl", "wb") as f:
    pickle.dump(gm_data, f)
