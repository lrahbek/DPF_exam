import os 
import pathlib
import pickle as pkl
import cloudpickle 
import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, RocCurveDisplay, average_precision_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

def split_features(feature_path, block_path ):
    """
    Load data (X, y) and split to train and test set
    """ 
    #X, y = load_prep_features(data_path) # load prepped data
    X, y, bill_ls = load_prepped_data(feature_path, block_path)
    lb = LabelBinarizer()                # define label binarizer (y)
    y = lb.fit_transform(y)              # fit and transform (y)
    y_classes = lb.classes_              # extract classes (y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=104, test_size=0.20, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test, y_classes, bill_ls

def preprocess_ct():
    """ 
    Defines column transformers for encoding and scaling features.    
    """
    categorical_transformer=Pipeline(steps=[('encoder', OneHotEncoder(sparse_output=False))])
    binary_transformer=Pipeline(steps=[('encoder', OneHotEncoder(sparse_output=False, drop="if_binary"))])
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    count_transformer = Pipeline(steps=[("scaler", RobustScaler())])
    text_transformer = Pipeline(steps=[('encoder', TfidfVectorizer(ngram_range = (1,2 ), 
                                                                   stop_words = "english", 
                                                                   min_df=0.001))])
    ct = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ["state"]),
            ("bin", binary_transformer, ["cha", "position"]),
            ("num", numeric_transformer, make_column_selector(dtype_include="float64")), 
            ("count", count_transformer, make_column_selector(dtype_include="int32")),
            ('text', text_transformer, "descr_prep")
        ], remainder='passthrough', force_int_remainder_cols = False, sparse_threshold=0)    
    return ct

def LR_pipeline_fit(X_train, y_train, out_path):
    """
    Defines a pipeline with the LogisticRegression Classifier (with the column transformer preprocessing),
    and fits it to the training data 
    """
    ct = preprocess_ct()
    LR = LogisticRegression(random_state=37, max_iter=10000, verbose=3, 
                            class_weight="balanced", solver="newton-cholesky")
    clf_pipeline_LR = Pipeline(steps=[("preprocessor", ct), ("LR_classifier", LR)])
    clf_pipeline_LR.fit(X_train, y_train)
    with open(out_path, 'wb') as file:
        cloudpickle.dump((clf_pipeline_LR, ct), file)
    return print("LogisticRegression finished fitting")


def MLP_pipeline_fit(X_train, y_train, out_path):
    """
    Defines a pipeline with the MLP Classifier (with the column transformer preprocessing),
    and fits it to the training data. (including gridsearch)
    """
    ct = preprocess_ct()
    MLP = MLPClassifier(early_stopping=True, random_state=37, max_iter=1000, verbose=3, activation="relu")
    parameter_grid = [{"solver": ['sgd', 'adam', 'lbfgs'], 
                       "learning_rate_init": [0.0001, 0.001, 0.01], 
                       "hidden_layer_sizes": [(100,), (125,), (150,)]}]
    grid_MLP = GridSearchCV(estimator = MLP, param_grid = parameter_grid, cv=5, verbose = 1,                          
                            scoring = "balanced_accuracy")
    clf_pipeline_MLP = Pipeline(steps=[("preprocessor", ct), ("classifier", grid_MLP)])
    clf_pipeline_MLP.fit(X_train, y_train)
    with open(out_path, 'wb') as file:
        cloudpickle.dump((clf_pipeline_MLP, ct), file)
    return print("MLPClassifier finished fitting")

def main():
    models_folder = "out/models/"
    feature_path = "data/preprocessed/features.pkl"
    block_path = "data/preprocessed/block_array.pkl"
    X_train, X_test, y_train, y_test, y_classes, bill_ls = split_features(
        feature_path = feature_path, block_path = block_path)
    LR_pipeline_fit(X_train, y_train, os.path.join(models_folder, "LR_fit.pkl"))
    MLP_pipeline_fit(X_train, y_train, os.path.join(models_folder, "MLP_fit.pkl"))


if __name__ == "__main__":
    main()