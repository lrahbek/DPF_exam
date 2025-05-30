import os 
import pathlib
import pickle as pkl
import cloudpickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import shap

def extract_SHAP(model_path, split_data_path, out_path):
    """
    The function loads in the fitted model pipeline and the split data. The best estimator from the 
    gridsearch and the feature names are extracted from the pipeline. A subset of the training and 
    testing data is transformed, to be used in the SHAP visualisations.
    A SHAP explanation object is created and the shap values, along with the remaing vars needed for
    crating SHAP visualtions are saved to file. 
    """
    with open(model_path, 'rb') as file:
        clf_pipeline_MLP  = pkl.load(file)
    with open(split_data_path, 'rb') as file:
        X_train, X_test, y_train, y_test,_,_= pkl.load(file)
    #model and feature names 
    MLP_best = clf_pipeline_MLP.named_steps.classifier.best_estimator_       
    feature_names = clf_pipeline_MLP.named_steps.preprocessor.get_feature_names_out().tolist()
    #subset and transform train and test data
    X_train_sub = X_train[0:800]
    X_test_sub = X_test[0:500]
    X_train_tr = clf_pipeline_MLP.named_steps.preprocessor.transform(X_train_sub)
    X_test_tr = clf_pipeline_MLP.named_steps.preprocessor.transform(X_test_sub)  
    #make predsictions on the transformed test data: 
    y_pred_sub = MLP_best.predict(X_test_tr)
    y_test_sub = y_test[0:500]
    #define the explainer and extract shap values 
    explainer = shap.KernelExplainer(MLP_best.predict_proba, X_train_tr, feature_names = feature_names)
    shap_values = explainer(X_test_tr)

    with open(out_path, 'wb') as file:
        cloudpickle.dump((shap_values, X_test_tr, y_test_sub, y_pred_sub), file)
    return print(f"shap values saved to {out_path}")


def main():
    model_path = 'out/models/MLP_fit.pkl'
    split_data_path = 'data/preprocessed/split_data.pkl'
    SHAP_path = 'out/models/shap_vals.pkl'
    
    extract_SHAP(model_path=model_path, split_data_path=split_data_path, out_path=SHAP_path)


if __name__ == "__main__":
    main()