from sklearn.model_selection import LeaveOneOut, KFold, RepeatedKFold, GridSearchCV
from jaqpotpy.descriptors.molecular import RDKitDescriptors, MordredDescriptors, MACCSKeysFingerprint, PubChemFingerprint
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from itertools import product
from sklearn.feature_selection import RFE, RFECV
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from kennard_stone import train_test_split as kennard_split
import io





def grid_search_cv(X_train, y_train, algorithm, param_grid: dict, metric : str,
                   cv: str, n_splits = None, n_repeats = None):
    
    metrics = ['neg_mean_squared_error',
               'neg_root_mean_squared_error',
               'r2']
    
    if cv.lower() == 'loo':
        cv = LeaveOneOut()
    elif cv.lower() == 'kfold':
        cv = KFold(n_splits = n_splits, shuffle=True, random_state=42)
    elif cv.lower() == 'repeatedkfold':
        cv = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = 42)
    else:
        raise ValueError(f"You choose {cv.lower()} which is invalid. Please choose 'loo', 'kfold', or 'repeatedkfold'.")

    if metric not in metrics:
        raise ValueError(f"Choose one of those metrics: {metrics}")

    pipeline = Pipeline([('scaler', MinMaxScaler()),
                     ('regressor', algorithm())])
    

    search = GridSearchCV(estimator = pipeline,
                          param_grid = param_grid,
                          cv = cv,
                          scoring = metric)
    
    search.fit(X_train, y_train)

    best_mean = search.cv_results_['mean_test_score'][search.best_index_]
    best_std = search.cv_results_['std_test_score'][search.best_index_]
    best_params = search.best_params_
    best_est = search.best_estimator_

    return best_mean, best_std, best_params, best_est


def grid_search_and_rfe(X_train, y_train, constant_features, algorithm, param_grid: dict, metric : str,
                   cv: str, threshold:int, n_splits = None, n_repeats = None):

    optim_dict = {}
    feats = X_train.columns
    mean_score = []
    total_feats = []
    n_features = len(feats)
    while len(feats) > 1:
        
        print(f'Optimizing for {len(feats)} features')
        bm, _ , bp ,b_est= grid_search_cv(X_train,
                                  y_train,
                                  algorithm=algorithm,
                                  param_grid=param_grid,
                                  metric = metric,
                                  cv = cv,
                                  n_splits = n_splits,
                                  n_repeats = n_repeats)
        

        optim_dict[str(len(feats))] = [bm, bp, feats]
        mean_score.append(-bm)
        total_feats.append(len(feats))

        bp = {key.replace('regressor__', ''): value 
                            for key, value in bp.items()}
        model = algorithm(**bp)
        model.fit(X_train, y_train)
        if len(feats) > threshold:
            rfe = RFE(estimator=model, 
               n_features_to_select= len(feats) - len(feats)//10, 
               step=len(feats)//10)
        else:
            rfe = RFE(estimator=model, 
               n_features_to_select= len(feats) - 1, 
               step=1)
        rfe.fit(X_train, y_train)
        # indices_to_keep = list(range(42))  # Generate indices from 0 to 41
        # rfe.support_[indices_to_keep] = True
        feats = rfe.get_feature_names_out()
        feats = list(set(feats.tolist() + constant_features))
        X_train = X_train[ feats ]
        print(len(feats))
        print(n_features)
        if len(feats) == n_features:
            break
        n_features = len(feats)

    sorted_dict = sorted(optim_dict.items(), key=lambda x:-x[1][0])

    plt.figure(figsize=(10, 5))  
    plt.plot(total_feats[-10:], mean_score[-10:], marker='o', linestyle='-', color='b')  
    plt.title('Model Performance by Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.grid(True)
    plt.xticks(total_feats[-10:], rotation=90)  
    plt.show()

    return sorted_dict



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
df = pd.read_csv('BCF_Modeling_dataset/final_dataset.csv')

x_data = df.drop(['log(BCF) (L/Kg)'], axis=1)
y_data = df['log(BCF) (L/Kg)']

# Eliminate correlated Mordred Descriptors (keep my descriptors in the dataset)
constant_features = ['Exposure Concentration (ug/L)', 'Exposure (days)', 
                     'Depuration (days)', 'Mixture', 'PFBS_mixture', 
                     'PFHxS_mixture', 'PFOS_mixture', 'PFBA_mixture', 
                     'PFPA_mixture', 'PFHxA_mixture', 'PFHpA_mixture', 
                     'PFOA_mixture', 'PFNA_mixture', 'PFDA_mixture', 
                     'PFUnA_mixture', 'PFDoA_mixture', 'PFTrDA_mixture',
                     'PFTeDA_mixture', 'PFHxPA_mixture', 'PFOPA_mixture',
                     'PFDPA_mixture', 'C6/C6_PFPiA_mixture', 'C6/C8_PFPiA_mixture', 
                     'C8/C8_PFPiA_mixture', 'C6/C10_PFPiA_mixture', 'C8/C10_PFPiA_mixture', 
                     'C6/C12_PFPiA_mixture', 'Species_Cyprinus carpio', 
                     'Species_Danio rerio', 'Species_Oncorhynchus mykiss', 
                     'Sex_both', 'Sex_female', 'Sex_male', 'Tissue_blood ', 
                     'Tissue_brain', 'Tissue_gill', 'Tissue_gonad', 'Tissue_liver', 
                     'Tissue_muscle', 'Tissue_ovary', 'Tissue_plasma', 'Tissue_whole body']

x_data_reduced, dropped_columns = feature_corellation(x_data.drop(columns = constant_features ), 0.9, 'pearson')

x_data = x_data[constant_features + x_data_reduced.columns.tolist() ]

# Split the datset to train and test set
x_train, x_test, y_train, y_test = kennard_split(x_data, y_data, test_size = 0.3, metric = 'euclidean')
x_train.columns = x_train.columns.astype(str)
x_test.columns = x_test.columns.astype(str)

algorithm = RandomForestRegressor
param_grid = {'regressor__random_state':[10],
              'regressor__n_estimators':[50, 75, 100],
              'regressor__max_depth':[5,10, 20]}

dc  = grid_search_and_rfe(x_train,
                        y_train,
                        constant_features=constant_features,
                        algorithm=algorithm,
                        param_grid=param_grid,
                        metric = 'neg_root_mean_squared_error',
                        cv = 'KFold',
                        n_splits=5,
                        threshold=42
                        )



# parameters for the best model
best_model_params = list(dc[0][1][1].values())
best_model_features = dc[0][1][2]
random_state = best_model_params[0]
n_estimators = best_model_params[1]
max_depth = best_model_params[2]
algorithm = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

#algorithm = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)

x_train_final = x_train[best_model_features]
x_test_final = x_test[best_model_features]
# Train the model with k-CV
q2, model = q2_cross_val(x_train_final, y_train, algorithm=algorithm,
                         cv = 'KFold', num_folds = 5, scaling = 'minmax')

scaler = MinMaxScaler()
y_pred_train = model.predict(scaler.fit_transform(x_train_final))
print(r2_score(y_train, y_pred_train))


y_pred_test = model.predict(scaler.transform(x_test_final))
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