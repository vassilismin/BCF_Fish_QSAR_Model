from jaqpotpy import jaqpot
from jaqpotpy.descriptors.molecular import RDKitDescriptors, MordredDescriptors, MACCSKeysFingerprint, PubChemFingerprint
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from kennard_stone import train_test_split as kennard_split
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
import xgboost as xgb
import matplotlib.pyplot as plt

#==================#
# CUSTOM FUNCTIONS #
#==================#
#=============================#
# Examine feature correlation #
#=============================#
# method = 'pearson'
# threshold = 0.9
# corr_matrix = x_data.corr(method).abs()  # Compute absolute values
# # Check only upper_tri for efficiency. k = 1 is for upper diagonal
# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
# # Drop first feature from a pair of two
# to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
# x_data = x_data.drop(to_drop, axis=1)

def feature_corellation(X_train: pd.core.frame.DataFrame,
                        threshold:float, method:str = 'pearson'):

    '''Function to check and drop correlated features. 
    Parameters:

       X_train: pd.DataFrame of input features and values of training data.

       threshold: absolute Value from which to drop corellated features. Takes value between 0 and 1

       method: correlation method: ('pearson', 'spearman', 'kendall'). Default is pearson
       !!Kendall takes long time to compute.

    Returns:

        reduced_X: pd.DataFrame of new number of features

        set(to_drop): Contains all the features drop. It can be used to drop them from a test dataset
    
    '''
    corr_matrix = X_train.corr(method).abs()  # Compute absolute values
    # Check only upper_tri for efficiency. k = 1 is for upper diagonal
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    # Drop first feature from a pair of two
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    reduced_X = X_train.drop(to_drop, axis=1)
    return reduced_X, set(to_drop)

#===========================#
# Cross validation Function #
#===========================#
def q2_cross_val(X_train, y_train, algorithm, cv, num_folds= 10, scaling = False):

    '''Function to compute q2_cross_validation for an sklearn model
    
    X_train: pd.DataFrame of features

    y_train: list or pd.Series with endpoint

    algorithm: SkLearn algorithm not initialized . RandomForestRegressor (!! not RandomForestRegressor)

    cv: cross validation method KFold or LeaveOneOut

    num_folds: Number of KFolds

    scaling: If scaling is implemented. Choose 'standard', 'minmax' or dont pass anything.

    **params: Parameters for the algorithm given. Must be as documented in sklearn exactly
    '''
    X_array = np.array(X_train)
    y_array = np.array(y_train)
    ytests = []
    ypreds = []
    feats = X_train.columns
    if cv.lower() == 'loo':
        cv = LeaveOneOut()
    elif cv.lower() == 'kfold':
        cv = KFold(num_folds, shuffle=True)
    elif cv.lower() == 'kennard_kfold':
        cv = Kennard_KFold(num_folds)
    else:
        raise ValueError(f"You choose {cv.lower()} which is invalid. Please choose 'loo', 'kfold', or 'kennard_kfold'.")
    # Iterate for the different train-validation folds
    for train_idx, test_idx in cv.split(X_array):

        X_train_cv, X_test_cv = X_array[train_idx], X_array[test_idx] #requires arrays
        y_train_cv, y_test_cv = y_array[train_idx], y_array[test_idx]

        model = algorithm #(**params)
        # Scaling calculated on train data. Performed on both train and test
        if not scaling:
            pass
        elif scaling.lower() == 'standard':
            scaler = StandardScaler()
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)
        elif scaling.lower() == 'minmax':
            scaler = MinMaxScaler()
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)
        else:
            raise ValueError(f"You choose {scaling.lower()}. Choose standard or minmax")
        # Fit the model on the train fold
        model.fit(X_train_cv, y_train_cv)
        # Predict on the test fold    
        y_pred = model.predict(X_test_cv)
        if cv == LeaveOneOut():
            ytests.append(y_test_cv)
            ypreds.append(y_pred)
        else:
            ytests.extend(y_test_cv)
            ypreds.extend(y_pred)
    # This is implemented based on r2 score only. No MAE or RMSE.
    q2 = r2_score(ytests, ypreds)

    return q2, model


#==================================#
# Plot Predictions vs Observations #
#==================================#

def plot_y_vs_y(y_observed, y_predicted, title):
    plt.scatter(y_observed, y_predicted, color='blue', label='Predicted vs Observed')
    plt.plot(y_observed, y_observed, color='red', linestyle='--', label='Ideal')
    plt.title(title)
    plt.xlabel('Observed log(BCF)')
    plt.ylabel('Predicted log(BCF)')
    plt.legend()
    plt.grid(True)


#=================#
# Data processing #
#=================#

# - Load csv data
df = pd.read_csv('BCF_Modeling_dataset/BCF_Dataset.csv')
# - remove rows with NAs in BCF
df.dropna(subset='log(BCF) (L/Kg)', inplace=True)
# - Keep specific columns for the training dataset
smiles, y_data = df['Smiles'], df['log(BCF) (L/Kg)']
# - Estimate descriptors
featurizer = MordredDescriptors()
features_df = pd.DataFrame(featurizer.featurize(smiles)) 

x = df[['Species', 'Sex', 'Tissue', 'Exposure Concentration (ug/L)', 
        'Exposure (days)',
       'Depuration (days)', 'Mixture', 'PFBS_mixture', 'PFHxS_mixture',
       'PFOS_mixture', 'PFBA_mixture', 'PFPA_mixture', 'PFHxA_mixture',
       'PFHpA_mixture', 'PFOA_mixture', 'PFNA_mixture', 'PFDA_mixture',
       'PFUnA_mixture', 'PFDoA_mixture', 'PFTrDA_mixture', 'PFTeDA_mixture',
       'PFHxPA_mixture', 'PFOPA_mixture', 'PFDPA_mixture',
       'C6/C6_PFPiA_mixture', 'C6/C8_PFPiA_mixture', 'C8/C8_PFPiA_mixture',
       'C6/C10_PFPiA_mixture', 'C8/C10_PFPiA_mixture', 'C6/C12_PFPiA_mixture']]

# Tranform the binary "Mixture" feature to integer values
x['Mixture'] = x['Mixture'].astype(int)
# Transform categorical features to one-hot-encoding
x = pd.get_dummies(x, columns=['Species', 'Sex', 'Tissue'])

# Merge estimated descriptors with the rest features of the dataset
x_data = pd.concat([x.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

# Eliminate correlated features
x_data, dropped_columns = feature_corellation(x_data, 0.9, 'pearson')

#print(x_data.isnull().any().to_string())
# Split the datset to train and test set
x_train, x_test, y_train, y_test = kennard_split(x_data, y_data, test_size = 0.3, metric = 'euclidean')
x_train.columns = x_train.columns.astype(str)
x_test.columns = x_test.columns.astype(str)

# Select model type
algorithm = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', max_depth = 5, random_state = 42)
#algorithm = xgb.XGBRegressor(n_estimators=1000, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=1.0)

# Train the model with k-CV
q2, model = q2_cross_val(x_train, y_train, algorithm=algorithm,
                         cv = 'KFold', num_folds = 5, scaling = False)


y_pred_train = model.predict(x_train)
print(r2_score(y_train, y_pred_train))


y_pred_test = model.predict(x_test)
print(r2_score(y_test, y_pred_test))



# Create plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for training set
plt.sca(axes[0])  # Select the first subplot
plot_y_vs_y(y_train, y_pred_train, title='Training Set')

# Plot for testing set
plt.sca(axes[1])  # Select the second subplot
plot_y_vs_y(y_test, y_pred_test, title='Testing Set')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#### Feature ranking and elimination
feature_ranking = model.feature_importances_
# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': x_data.columns, 'Importance': feature_ranking})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df[importance_df['Importance'] != 0]

# Create a Random Forest Classifier as the base model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Specify the number of features to select using RFE
num_features_to_select = 20

# Initialize RFE with the model and the number of features to select
rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)

# Fit RFE to the training data
rfe.fit(x_train, y_train)

# Get the ranking of each feature
feature_ranking = rfe.ranking_

# Get the selected features
selected_features = np.where(feature_ranking == 1)[0]

# Visualize the feature ranking
plt.figure(figsize=(10, 6))
plt.title("RFE - Feature Ranking")
plt.xlabel("Feature Index")
plt.ylabel("Ranking")
plt.bar(range(len(feature_ranking)), feature_ranking)
plt.show()

# Print the selected features
print("Selected Features:", selected_features)

q2_rfe, model_rfe = q2_cross_val(x_train[:, 1], y_train, algorithm=algorithm,
                         cv = 'KFold', num_folds = 5, scaling = False)

