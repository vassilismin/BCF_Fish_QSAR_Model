df = openxlsx::read.xlsx('/Users/vassilis/Documents/GitHub/BAC_BCF_models/Albumin_binding_model/data/Albumin_binding_data.xlsx')

congeners <- unique(df$Ligand.Name)
 
# openxlsx::write.xlsx(data.frame(congeners), '/Users/vassilis/Documents/GitHub/BAC_BCF_models/Albumin_binding_model/smiles_list.xlsx') 

df = df[-which(df$Ligand.Name %in% c('br-PFOS')),]

smiles_list <- openxlsx::read.xlsx('/Users/vassilis/Documents/GitHub/BAC_BCF_models/Albumin_binding_model/data/smiles_list.xlsx')

for (i in 1:dim(df)[1]) {
  pfas_name <- df$Ligand.Name[i]
  take_smiles <- smiles_list[which(smiles_list$congeners == pfas_name) , 3]
  df$Canonical.SMILES[i] <- take_smiles
}

keep_columns <- c('Ligand.Name', "Canonical.SMILES", "Albumin.source.organism", "Albumin.concentration,.μM",
                  'Method', 'Scatchard.equation', 'Temperature,.K')

openxlsx::write.xlsx(df, '/Users/vassilis/Documents/GitHub/BAC_BCF_models/Albumin_binding_model/data/Albumin_binding_data_Updated.xlsx')

x_data <- df[,keep_columns]
y_data <- df$Ka.in.M.scale

write.csv(x_data, '/Users/vassilis/Documents/GitHub/BAC_BCF_models/Albumin_binding_model/data/x_data.csv', row.names = FALSE)
write.csv(y_data, '/Users/vassilis/Documents/GitHub/BAC_BCF_models/Albumin_binding_model/data/y_data.csv', row.names = FALSE)
