# Bioconcentration Factor Prediction for PFAS

The bioconcentration factor (BCF) measures the accumulation of a substance in aquatic organisms relative to its concentration in water (BCF = C_organism / C_water). This QSAR model predicts the BCF of various PFAS compounds in fish species using data from seven studies, encompassing 283 instances for 25 PFAS congeners.

The dataset includes measurements for three fish species—*Danio rerio*, *Oncorhynchus mykiss*, and *Cyprinus carpio*—and incorporates features such as exposure concentration, duration, sex, tissue type, and whether the organisms were exposed to PFAS mixtures. Additionally, molecular descriptors for PFAS were computed using Mordred based on SMILES representations.

Categorical features like species, sex, and tissue type were digitized with one-hot encoding, and binary variables were converted for model compatibility. The model was developed and trained using robust preprocessing pipelines to ensure accurate and reliable predictions.

## Jaqpot Service 
This Qsar model haws been deployed on Jaqpot Application as an online service. You can find this and run online this model at https://app.jaqpot.org/dashboard/models/1976/description
