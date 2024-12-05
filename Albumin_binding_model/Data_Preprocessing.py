from jaqpotpy.descriptors.molecular import RDKitDescriptors, MordredDescriptors, MACCSKeysFingerprint, PubChemFingerprint
import pandas as pd
import numpy as np

# This script estimates the descriptors and creates the final dataset

#=================#
# Data processing #
#=================#

# - Load csv data
x_data = pd.read_csv('data/x_data.csv')

mean_temperature = x_data['Temperature,.K'].mean()
x_data['Temperature,.K'] = x_data['Temperature,.K'].fillna(mean_temperature)

smiles = x_data['Canonical.SMILES']
# - Estimate descriptors
featurizer = MordredDescriptors()
computational_descriptors = pd.DataFrame(featurizer.featurize(smiles)) 

# Tranform catecorical descriptors to binary with one-hot encoding
x_data = pd.get_dummies(x_data, columns = ['Albumin.source.organism', 'Method'])
x_data['Scatchard.equation'] = x_data['Scatchard.equation'].astype(int)
x_data_final = pd.concat([x_data, computational_descriptors], axis=1)

x_data_final.drop(['Ligand.Name', 'Canonical.SMILES'], axis=1, inplace=True)

# Transform y data to log10 values
y_data = pd.read_csv('data/y_data.csv')
y_data_final = np.log10(y_data)

y_data_final.columns = ['LogKa']

# Create a dataframe with x_data_final and y_data_final, that will be used for the training of the model
final_df = pd.concat([x_data_final, y_data_final], axis=1)
final_df.to_csv('data/Final_df.csv', index = False)
