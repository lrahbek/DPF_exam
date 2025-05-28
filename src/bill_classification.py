import os 
import pathlib
import pickle as pkl
import cloudpickle 
import pandas as pd
import numpy as np
import re
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier


def load_prepped_data(feature_path, block_path):
    """ 
    Load prepped data (features, X and y) and block array (with block counts). Create df from 
    block_array and split each block into support, neutral and oppose columns. append the resulting
    df to X and return. (also returned: the list of bill ids in same order as X df)
    """
    with open(feature_path, 'rb') as file:
        X, y = pkl.load(file)        
    with open(block_path, 'rb') as file:
        block_array, block1_ls, bill_ls = pkl.load(file)
        
    block_col_names = np.char.add("block1_", block1_ls) #define column names 
    block_df = pd.DataFrame.from_records(block_array, columns=block_col_names)
    block_df_split = pd.DataFrame()                     #create empty df for split cols
    #populate empty df
    for col in block_df.columns:
        block_df_split[[(col+"_s"), (col+"_n"), (col+"_o")]] = pd.DataFrame(block_df[col].to_list())
    #add lobby column(if no positions on the bill the row sum will be 0 i.e. no_lobby)
    block_df_split["position"] = np.where(block_df_split.sum(axis=1)>0, 'yes', 'no')
    #join X and block df 
    X = X.join(block_df_split)
    return X, y, bill_ls

def split_features(feature_path, block_path, split_data_path):
    """
    Load data (X, y), perform random undersampling and split to train and test set
    """ 
    #X, y = load_prep_features(data_path) # load prepped data
    X, y, bill_ls = load_prepped_data(feature_path, block_path)
    y = y.ravel() #change shape of y
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=37)#resample
    X_rs, y_rs = rus.fit_resample(X,y)
    lb = LabelBinarizer()                # define label binarizer (y)
    y_rs = lb.fit_transform(y_rs)              # fit and transform (y)
    y_classes = lb.classes_              # extract classes (y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_rs, y_rs, random_state=104, test_size=0.20, shuffle=True, stratify=y_rs)
    
    with open(split_data_path, 'wb') as file:
        pkl.dump((X_train, X_test, y_train, y_test, y_classes, bill_ls), file)

    return X_train, X_test, y_train, y_test, y_classes, bill_ls

def preprocess_ct():
    """ 
    Defines column transformers for encoding and scaling features.    
    """
    categorical_transformer=Pipeline(steps=[('encoder', OneHotEncoder(sparse_output=False))])
    binary_transformer=Pipeline(steps=[('encoder', OneHotEncoder(sparse_output=False, drop="if_binary"))])
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    count_transformer = Pipeline(steps=[("scaler", RobustScaler())])

    ct = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ["state"]),
            ("bin", binary_transformer, ["cha", "position"]),
            ("num", numeric_transformer, make_column_selector(dtype_include="float64")), 
            ("count", count_transformer, make_column_selector(dtype_include="int32")),
        ], remainder='drop', force_int_remainder_cols = False, sparse_threshold=0)    
    return ct


def MLP_pipeline_fit(X_train, y_train, out_path):
    """
    Defines a pipeline with the MLP Classifier (with the column transformer preprocessing),
    and fits it to the training data. (including gridsearch)
    """
    ct = preprocess_ct()
    MLP = MLPClassifier(early_stopping=True, random_state=37, max_iter=1000, verbose=1, 
                        activation='relu', solver='adam', tol=0.0001, n_iter_no_change = 10, 
                        validation_fraction=0.1, hidden_layer_sizes=(100,))

    parameter_grid = [{'alpha':[0.001, 0.0001, 0.00001],
                       'learning_rate_init': [0.0001, 0.001, 0.01]}]
    grid_MLP = GridSearchCV(estimator = MLP, param_grid = parameter_grid, cv=3, verbose = 1,                          
                            scoring = "balanced_accuracy")
    clf_pipeline_MLP = Pipeline(steps=[("preprocessor", ct), ("classifier", grid_MLP)])
    clf_pipeline_MLP.fit(X_train, y_train)

    with open(out_path, 'wb') as file:
        cloudpickle.dump(clf_pipeline_MLP, file)
    return print("MLPClassifier finished fitting")


def main():
    models_folder = "out/models/"
    feature_path = "data/preprocessed/features.pkl"
    block_path = "data/preprocessed/block_array.pkl"
    split_data_path = "data/preprocessed/split_data.pkl"

    X_train, X_test, y_train, y_test, y_classes, bill_ls = split_features(
        feature_path = feature_path, block_path = block_path, split_data_path=split_data_path)

    MLP_pipeline_fit(X_train, y_train, os.path.join(models_folder, "MLP_fit.pkl"))

if __name__ == "__main__":
    main()