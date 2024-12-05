import pandas as pd

# This script estimates the descriptors and creates the final dataset

# =================#
# Data processing #
# =================#

# - Load csv data
df = pd.read_csv("BCF_Modeling_dataset/BCF_Dataset.csv")
# - remove rows with NAs in BCF
df.dropna(subset="log(BCF) (L/Kg)", inplace=True)
# - Keep specific columns for the training dataset
y_data = df["log(BCF) (L/Kg)"]

x = df[
    [
        "Smiles",
        "Species",
        "Sex",
        "Tissue",
        "Exposure Concentration (ug/L)",
        "Exposure (days)",
        "Depuration (days)",
        "Mixture",
        # "PFBS_mixture",
        # "PFHxS_mixture",
        # "PFOS_mixture",
        # "PFBA_mixture",
        # "PFPA_mixture",
        # "PFHxA_mixture",
        # "PFHpA_mixture",
        # "PFOA_mixture",
        # "PFNA_mixture",
        # "PFDA_mixture",
        # "PFUnA_mixture",
        # "PFDoA_mixture",
        # "PFTrDA_mixture",
        # "PFTeDA_mixture",
        # "PFHxPA_mixture",
        # "PFOPA_mixture",
        # "PFDPA_mixture",
        # "C6/C6_PFPiA_mixture",
        # "C6/C8_PFPiA_mixture",
        # "C8/C8_PFPiA_mixture",
        # "C6/C10_PFPiA_mixture",
        # "C8/C10_PFPiA_mixture",
        # "C6/C12_PFPiA_mixture",
    ]
]

# Tranform the binary "Mixture" feature to integer values
# x["Mixture"] = x["Mixture"].astype(int)
# Transform categorical features to one-hot-encoding
# x = pd.get_dummies(x, columns=["Species", "Sex", "Tissue"])

# Merge estimated descriptors with the rest features of the dataset
# x_data = x.reset_index(drop=True)

final_df = pd.concat([x.reset_index(drop=True), y_data.reset_index(drop=True)], axis=1)
print(final_df.head())
# write to csv
final_df.to_csv("BCF_Model_Jaqpot/jaqpot_dataset.csv", index=False)
