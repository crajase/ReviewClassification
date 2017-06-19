import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import os
import re


def readNonNullData(file_csv):
    df = pd.read_csv(file_csv)
    headers = list(df.columns.values)
    headers = [header for header in headers if not header.startswith("Unnamed")]

    for head in headers:
        df = df[df[head].notnull()]
    return df


def DFToFeatureX(feature_model, df, x_label):
    featuresX = []
    df = df[x_label]
    for string in df:
        string = re.sub("[^a-zA-Z?!]", " ", str(string))
        # TODO: Clean by removing stopwords
        words = [w.lower() for w in string.strip().split() if len(w) >= 3]
        featuresX.append(feature_model.infer_vector(words))
    return pd.DataFrame(featuresX)


# feature is a 2D list, where row represents data samples and col represent features
def standardize(x_df_features, featureScale=True, makePolynomial=True, d=1):
    # normalization and feature scaling
    if featureScale:
        print("FeatureScale")
        x_df_features = (x_df_features - x_df_features.mean()) / (x_df_features.max() - x_df_features.min())

    # Try d degree polynomial
    if makePolynomial:
        print("Poly of degree:", d)
        x_df_features = x_df_features**d

    return x_df_features


# splits the data into training and test set.
# TODO: To add cross validation set
def splitData(features, y_output, num_test):
    train_length = len(features) - num_test
    x_train = features[0:train_length]
    x_test = features[train_length:]
    y_train = y_output[0:train_length]
    y_test = y_output[train_length:]
    return x_train, y_train, x_test, y_test


def analyseFeature(feature, y_output):
    return feature
    