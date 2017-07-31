import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import os
import re
import operator
from sklearn.decomposition import PCA


def readNonNullData(file_csv, x_label, y_label):
    df = pd.read_csv(file_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    headers = list(df.columns.values)
    headers = [header for header in headers if not header.startswith("Unnamed")]
    print(headers)
    df = df[headers]
    df = df[df[x_label].notnull()]
    df = df[df[y_label].notnull()]
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


def filterFeatures(features, column_list):
    features = features[column_list]
    return features


# feature is a 2D list, where row represents data samples and col represent features
def standardize(x_df_features, featureScale=True, makePolynomial=True, d=1):
    # normalization and feature scaling
    if featureScale:
        print("FeatureScale")
        x_df_features = (x_df_features - x_df_features.mean()) / (x_df_features.max() - x_df_features.min())
        x_df_features = x_df_features * 2
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


def PCA_reduce_feature(features, num_comp):
    pca = PCA(num_comp)
    return pd.DataFrame(pca.fit_transform(features))


def splitPosAndNeg(features, y_output):
    if len(features) != len(y_output):
        return None
    features = features.values.tolist()
    y_output = y_output.values.tolist()
    features_pos = []
    features_neg = []
    for i in range(len(y_output)):
        if y_output[i][0] == 0:
            features_neg.append(features[i])
        else:
            features_pos.append(features[i])
    return pd.DataFrame(features_pos), pd.DataFrame(features_neg)


def analyseFeature(features_pos, features_neg, n_feature):
    mean_pos = features_pos.mean()
    var_pos = ((features_pos - mean_pos)**2).mean()
    sd_pos = var_pos**0.5
    mean_neg = features_neg.mean()
    var_neg = ((features_neg - mean_neg)**2).mean()
    sd_neg = var_neg**0.5

    diff_mean = (mean_pos - mean_neg).abs()
    diff_sd = (sd_pos + sd_neg).abs()
    print("Pos Mean: ", mean_pos.mean())
    print("Neg Mean: ", mean_neg.mean())
    dist = (var_pos + var_neg) - diff_mean
    # dist = (sd_pos + sd_neg) - diff_mean
    analyse = (dist > 0).values
    dist_np = dist.values
    dist_vec = {i: dist_np[i] for i in range(len(dist_np))}
    print(type(dist_vec))
    dist_sorted = sorted(dist_vec.items(), key=operator.itemgetter(1))
    # print(dist_vec)
    # ret = [i for i in dist_vec.keys() if dist_vec[i] < 0]
    ret = [dist_sorted[i][0] for i in range(n_feature)]
    print("Dist: ", dist.mean())
    print("Cont: ", np.count_nonzero(analyse))
    print(ret)
    return ret


def shuffle_and_distribute(features, y_output, testPerc):
    feature_pos, feature_neg = splitPosAndNeg(features, y_output)
    pos_Columns = feature_pos.columns
    neg_Columns = feature_neg.columns
    feature_pos['target'] = 1
    feature_neg['target'] = 0
    train_len_pos = int((100 - testPerc) * len(feature_pos) / 100)
    train_len_neg = int((100 - testPerc) * len(feature_neg) / 100)

    train_data = feature_pos[0:train_len_pos]
    test_data = feature_pos[train_len_pos:]
    train_data = train_data.append(feature_neg[0:train_len_neg], ignore_index=True)
    test_data = test_data.append(feature_neg[train_len_neg:], ignore_index=True)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    return train_data[pos_Columns], train_data.loc[:, ['target']], test_data[neg_Columns], test_data.loc[:, ['target']]
