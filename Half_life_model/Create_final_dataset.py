from jaqpotpy.descriptors.molecular import RDKitDescriptors, MordredDescriptors, MACCSKeysFingerprint, PubChemFingerprint
import pandas as pd

# This script estimates the descriptors and creates the final dataset

#=================#
# Data processing #
#=================#

# - Load csv data
df = pd.read_csv('half_life_data.csv')

features_names = df.columns
# - remove rows with NAs in 
df.dropna(subset='half.life..days.', inplace=True)

# - drop rows withoud smiles representation
df.dropna(subset='Smiles', inplace=True)
# - Keep specific columns for the training dataset
smiles, y_data = df['Smiles'], df['half.life..days.']
# - Estimate descriptors
featurizer = MordredDescriptors()
features_df = pd.DataFrame(featurizer.featurize(smiles)) 

x = df[['tissue', 'species', 'sex', 'route.of.administration',
       'mode.of.administration', 'adult']]

# Transform 'adult' feature to 0s and 1s
x['adult'] = x['adult'].astype(int)

# Transform categorical features to one-hot-encoding
x = pd.get_dummies(x, columns=['tissue', 'species', 'sex', 'route.of.administration',
       'mode.of.administration'])

# Merge estimated descriptors with the rest features of the dataset
x_data = pd.concat([x.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

final_df = pd.concat([x_data.reset_index(drop=True), y_data.reset_index(drop=True)], axis=1)

# write to csv
final_df.to_csv('final_dataset.csv', index=False)