import os 
import pathlib
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

def load_prep_features(path):
    """ Load preprocessed data """
    with open(path, 'rb') as file:
        features = pkl.load(file)
    return features

def encode_y(features_df, y_col:str):
    """ Use labelencoder (sklearn) to encode outcome var, return y and class names"""
    le = LabelEncoder()
    y = le.fit_transform(features_df[y_col])
    y_classes = le.classes_
    return y, y_classes

def ordinal_years_encoder(X_df, X_feature, years_list):
    """ 
    Uses sklearns OrdinalEncoder to encode years as ordered categorical variables. Takes the df
    with the feature (X_df), the column names (X_feature) and an ordered list of the categories
    as a list. Returns the categories (names) and transformed features. 
    """
    enc = OrdinalEncoder(categories=years_list, dtype="int32")
    X_transformed = enc.fit_transform(X_df[[X_feature]].to_numpy())
    names = enc.categories_
    return X_transformed, names

def categroical_encoder(X_df, X_feature:str, enc_type:str):
    """
    Encoded categorical feature. Feature is given by a dataframe (X_df) and a given column 
    name (X_feature). The type of encodding is given by a string (enc_type). Either 'onehot' or
    'multi' (resulting in using sklearn OneHotEncoder or CountVectoriser respectively). The 
    function returns the transformed features. 
        onehot is used for 'state'
        multi is used for 'ncsl_topics', 'ncsl_metatopics', 
    """
    assert enc_type in ["onehot", "multi"], "enc_type should be either 'onehot' or 'multi'"
    if enc_type == "onehot":
        enc = OneHotEncoder(sparse_output=False)
        X_transformed = enc.fit_transform(X_df[[X_feature]].to_numpy())
        names = enc.get_feature_names_out([X_feature])
    elif enc_type == "multi":
        enc = CountVectorizer(analyzer=lambda lst: lst)
        X_transformed = enc.fit_transform(X_df[X_feature]).toarray()
        names = enc.get_feature_names_out()     
    return X_transformed, names

def encode_blocks(X_df, col_blocks:list):
    """ 
    Fits a sklearn Countvectoriser to the blocks that have positions on the bills. Takes the df with
    the features (X_df) and the names of the columns with blocks as a list (col_blocks). Returns the
    fitted encoder and the feature names
    """
    enc_blocks = CountVectorizer(analyzer=lambda lst: lst)
    enc_blocks.fit(np.concatenate((X_df[col_blocks[0]].to_numpy(), 
                                   X_df[col_blocks[1]].to_numpy(), 
                                   X_df[col_blocks[2]].to_numpy())))
    names_blocks = enc_blocks.get_feature_names_out()
    return enc_blocks, names_blocks

def transform_block_counts(X_df, blocks_col:str, counts_col:str, fitted_encoder):
    """ 
    Uses a fitted encoder to transform block features, then the respective count data is used inputted
    to to features. It returns the transformed feature. It takes the df with the features (X_df), the
    column name with block assignments (blocks_col), the column name with the count data, and the 
    fitted encoder.  
    """
    X_blocks = fitted_encoder.transform(X_df[blocks_col]).toarray()
    X_counts = X_df[counts_col]
    
    for i, (row_blocks, row_counts) in enumerate(zip(X_blocks, X_counts), start=0):
        if type(row_counts) == list:
            inds = np.asarray(row_blocks==1).nonzero()[0]
            for j, (index, count) in enumerate(zip(inds, row_counts), start=0):
                row_blocks[index] = count  
    return X_blocks


def main():
    features = load_prep_features('../data/preprocessed/features.pkl')
    #get X and y and split into train and test: 
    y, y_classes = encode_y(features, "pass")
    feature_names = ["state", "ncsl_topics", "ncsl_metatopics", "bill_year", "lobbied", 
                     "neut_blocks", "opp_blocks", "sup_blocks", "neut_counts", "opp_counts", "sup_counts"]
    X = features[feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=104, test_size=0.20, shuffle=True, stratify=y)



if __name__ == "__main__":
    main()