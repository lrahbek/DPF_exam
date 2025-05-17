## load packages
import os 
import pathlib
import gdown
import sklearn
import pyarrow
import pandas as pd
import numpy as np
import itertools
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


def encode_multi_features(dataframe, col_name, split_chr):
    """ 
    Encode features with different number of feature representations, using CountVectorizer. The function takes the 
    dataframe and column name that holds the features (as a single string), splits them on the given value and returns
    an array with the transformed features and an array of the feature names.
    """
    features = dataframe[col_name].str.split(split_chr)
    enc = CountVectorizer(analyzer=lambda lst: lst)
    transformed_features = enc.fit_transform(features).toarray()
    feature_names = enc.get_feature_names_out()    
    return transformed_features, feature_names


def encode_categorical(dataframe, col_names):
    """ 
    Encode categorical features with OneHotEncoder. The function takes a dataframe and list of column names that should
    be encoded. It returns an array of the transformed features and feature names
    """
    features = dataframe[col_names].to_numpy()
    enc = OneHotEncoder()
    transformed_features = enc.fit_transform(features)
    feature_names = enc.get_feature_names_out(col_names) 
    return transformed_features, feature_names