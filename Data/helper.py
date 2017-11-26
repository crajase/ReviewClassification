import pandas as pd
import numpy as np
import data_clean as dc


def readData(file_csv, x_label):
    df = pd.read_csv(file_csv)
    # df = df.sample(frac=1).reset_index(drop=True)
    headers = list(df.columns.values)
    headers = [header for header in headers if not header.startswith("Unnamed")]
    print(headers)
    df = df[headers]
    df = df[df[x_label].notnull()]
    return df


def getNotNull(df, y_label):
    df = df[df[y_label].notnull()]
    return df


def NotNullAllLabels(df, y_labels):
    for y_label in y_labels:
        df = df[df[y_label].notnull()]
    return df


def split(df, perc):
    train_cnt = int(df.shape[0] * perc / 100)
    df_train = df[:train_cnt]
    df_test = df[train_cnt:]
    return df_train, df_test


def stratified_sampling(df, Y, perc):
    print("------------Performing Stratified Sampling------------------")
    pos = (Y != 0)
    neg = (Y == 0)
    train_pos, test_pos = split(df[pos], perc)
    train_neg, test_neg = split(df[neg], perc)
    train = train_pos.append(train_neg, ignore_index=True)
    test = test_pos.append(test_neg, ignore_index=True)
    return train.sample(frac=1).reset_index(drop=True), test.sample(frac=1).reset_index(drop=True)


def DFToFeatureX_word2mat(feature_model, df, x_label, idf_model=None, max_len=500):
    # TODO: Check each step for correctness
    featuresX = []
    vec_len = 300
    df = df[x_label]
    for string in df:
        # TODO: Clean by removing stopwords
        words = dc.get_clean_wrd_list(string)
        if(idf_model is not None):
            word_idf = idf_model.calc_tf_idf(words)
        else:
            word_idf = None
        words_vec = []
        for word in words:
            if(word in feature_model):
                if(word_idf is None):
                    feat = [ft for ft in feature_model[word]]
                else:
                    feat = [ft * word_idf[word] for ft in feature_model[word]]
            else:
                feat = np.zeros(vec_len)

            words_vec.append(feat)

        if(len(words_vec) > 0):
            sent_vec = np.array(words_vec)
            featuresX.append(sent_vec)
        else:
            featuresX.append(np.zeros((1, vec_len)))

    featuresX = [np.pad(sent_vec1, ((0, max_len - sent_vec1.shape[0]), (0, 0)), mode='constant', constant_values=0) for sent_vec1 in featuresX]
    return np.asarray(featuresX)


def DFToFeatureX_doc2vec(feature_model, df, x_label):
    featuresX = []
    df = df[x_label]
    for string in df:
        words = dc.get_clean_wrd_list(string)
        featuresX.append(feature_model.infer_vector(words))
    return pd.DataFrame(featuresX)


def DFToFeatureX_word2vec(feature_model, df, x_label, idf_model=None):
    # TODO: Check each step for correctness
    featuresX = []
    vec_len = 300  # feature_model.layer1_size
    df = df[x_label]
    for string in df:
        # TODO: Clean by removing stopwords
        words = dc.get_clean_wrd_list(string)
        if(idf_model is not None):
            word_idf = idf_model.calc_tf_idf(words)
        else:
            word_idf = None
        words_vec = []
        for word in words:
            if(word in feature_model):
                if(idf_model is None):
                    feat = feature_model[word]
                else:
                    feat = [ft * word_idf[word] for ft in feature_model[word]]
                words_vec.append(feat)
        if(len(words_vec) > 0):
            sent_vec = np.array(words_vec).mean(axis=0)
            featuresX.append(sent_vec)
        else:
            featuresX.append(np.zeros(vec_len))
    return pd.DataFrame(featuresX)

