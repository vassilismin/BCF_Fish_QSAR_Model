df = openxlsx::read.xlsx('/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/BCF_Dataset.xlsx')

# Number of instances
n_instances <- nrow(df)


# Number of congeners
n_congeners <- length(unique(df$Substance))
congeners <- unique(df$Substance)
# Number of studies
n_studies <- length(unique(df$Study))

# Different tissues
tissues <- unique(df$Tissue)

min(df$`Exposure.Concentration.(ug/L)`)
max(df$`Exposure.Concentration.(ug/L)`)

colnames(df)
sub_df <- df[,c("Substance", "Abbreviation", "CAS")]
sub_df <- sub_df[!duplicated(sub_df$Abbreviation),]
dim(sub_df)

pfas_list <- sub_df
write.csv(pfas_list, '/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/PFAS_list.csv')

studies_list <- df[!duplicated(df$Study) , c("Study", "DOI")]
write.csv(studies_list, '/Users/vassilis/Documents/GitHub/BAC_BCF_models/BCF_Modeling_dataset/studies_list.csv')
