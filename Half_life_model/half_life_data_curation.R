setwd('/Users/vassilis/Documents/GitHub/BAC_BCF_models/Half_life_model')

df <- read.csv('Half-life_dataset_mammals_copy_vassilis.xlsx - Sheet1.csv')

# drop the rows from the Arnot et al., 2014 study (half-lives from qsar model)
df <- df[-which(df$reference == 'Arnot et al., 2014'),]

# drop the rows from the Yao et al., 2023 study (half-life estimated for infants)
df <- df[-which(df$reference == 'Yao et al., 2023'),]

features_names <- colnames(df)

unique(df$species)

# replace all Species to lower letters
df$species <- tolower(df$species) 

tissues <- unique(df$tissue)

# replace blood serum with blood
df$tissue[which(df$tissue == "blood serum")] <- 'blood'

# replace "blood (whole)" with blood
df$tissue[which(df$tissue == "blood (whole)")] <- 'blood'

# replace "serum" with blood
df$tissue[which(df$tissue == "serum")] <- 'blood'

# replace "back fat" with fat
df$tissue[which(df$tissue == "back fat")] <- 'fat'

# replace "IP fat" with fat
df$tissue[which(df$tissue == "IP fat")] <- 'fat'

# replace' whole body (blood/urine combined)' with 'whole body'
df$tissue[which(df$tissue == "whole body (blood/urine combined)")] <- 'blood'

# replace 'serum,urine' with 'whole body'
df$tissue[which(df$tissue == "serum,urine")] <- 'blood'

# replace 'blood plasma' with 'whole body'
df$tissue[which(df$tissue == "blood plasma")] <- 'blood'

# replace 'blood, urine, milk' with 'whole body'
df$tissue[which(df$tissue == "blood, urine, milk")] <- 'blood'

# replace 'heifer' with 'cattle'
df$species[which(df$species == "heifer")] <- 'cattle'

# replace 'steer' with 'cattle'
df$species[which(df$species == "steer")] <- 'cattle'

human_rows = sum(df$species == 'human')

species_data = data.frame(matrix(c(sum(df$species == 'human'), 
                                   sum(df$species == 'monkey'),
                                   sum(df$species == 'mouse'),
                                   sum(df$species == 'rat'),
                                   sum(df$species == 'cattle'),
                                   sum(df$species == 'pig'))))
colnames(species_data) <- 'Rows'
rownames(species_data) <- c('human', 'monkey', 'mouse', 'rat', 'cattle', 'pig')


# routes of administration
unique(df$route.of.administration)

# replace 'occupational' with 'pollution'
df$route.of.administration[which(df$route.of.administration == "occupational")] <- 'pollution'

# mode of administration
unique(df$mode.of.administration)

# replace 'repeated daily' with 'repeated'
df$mode.of.administration[which(df$mode.of.administration == "repeated daily")] <- 'repeated'

# replace "continuous " with 'continuous'
df$mode.of.administration[which(df$mode.of.administration == "continuous ")] <- 'continuous'

write.csv(df, 'half_life_data.csv', row.names = FALSE)


unique(df$mode.of.administration)
