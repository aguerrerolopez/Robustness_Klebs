from pyteomics import mzml
import os
from preprocess import SpectrumObject
import pandas as pd


path_folder = "data/RyC/mzml"

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
            id = mzml_file.split("/")[-1].split(".")[0].split("_")[0:2]
            # if the first element is a number, then both elements are united with "-", otherwise, they are united with nothing ""
            if id[0].isdigit():
                id = "-".join(id)
            else:
                id = "".join(id)
            id_list.append(id)


path_excel = "data/RyC/DB_conjunta.xlsx"

df = pd.read_excel(path_excel)

# Now, get from df_final the ones that match "Nº Espectro" with id_list
df["Número de muestra"] = df["Número de muestra"].astype(str)
df_final = df[df["Número de muestra"].isin(id_list)]

# Which antibiotic do we want to check? In this case, CEFEPIME.
antibiotic = "CEFEPIME"
columns_to_get = ["Número de muestra", antibiotic, antibiotic + ".1"]
df_final = df_final[columns_to_get]


# contruct the final X and Y datasets
X_int = []
X_mz = []
Y = []
for id in id_list:
    row = df_final[df_final["Número de muestra"] == id]
    if row.empty:
        continue
    row = row.iloc[0]
    X_int.append(spectra[id_list.index(id)].intensity)
    X_mz.append(spectra[id_list.index(id)].mz)
    y_raw = row[antibiotic + ".1"]
    # subtitute R and I by 1 and S by 0
    y = 1 if y_raw == "R" or y_raw == "I" else 0
    Y.append(y)


# Save the data in a pkl
import pickle

gm_data = {
    "X_int": X_int,
    "X_mz": X_mz,
    "Y": Y,
}

with open("ryc_data.pkl", "wb") as f:
    pickle.dump(gm_data, f)
