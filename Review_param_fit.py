import helper as hlp
import pandas as pd
import numpy as np
import idf
import ff_model_new as ff_n
from datetime import datetime
import random
import os
import convff
import gensim
import math

data_csv_file = "TrimedData.csv"
y_labels = ["Praise", "Problem", "Solution", "Mitigation", "Neutrality", "Localization", "Summary"]
x_label = "Comments"
df_whl = hlp.readData(data_csv_file, x_label)

num_samples = 25
feature_dim = 300
which_feat = 2
idf_model = None
if(which_feat == 1):
    feature_model = gensim.models.Doc2Vec.load("comments2vec.d2v")
    feature_name = "Doc2Vec"
elif(which_feat == 2):
    feature_model = gensim.models.Word2Vec.load("cmtWord2vec" + str(feature_dim) + ".d2v")
    feature_name = "Word2Vec"
else:
    idf_model = idf.Idf.load("idf_model.json")
    feature_name = "Word2VecIDF"

model_params = ["Model", "No of Layers", "No of Nodes", "lambda", "epoch", "Learning Rate"]
feature_params = ["Feature Used", "Feature Dimension"]
output_attibute_list = ["Train Cost", "Test Cost", "Accuracy", "True Pos", "False Pos", "False Neg", "True Neg", "F1-Score"]
result_column_list = feature_params + model_params + output_attibute_list
hyper_params_dict = {}
hyper_params_lambdas = 0
hyper_params_learning_rt = [0.01, 1000, 0.96]
hyper_params_nodes = [200, 75, 25]
hyper_params_n = 3
epoch = 15

min_node_cnt = 10


def select_random_nodes():
    global hyper_params_n, hyper_params_nodes
    hyper_params_n = random.randint(1, 5)
    mean = feature_dim / hyper_params_n
    hyper_params_nodes = [int(np.random.normal((hyper_params_n - i) * mean, (25 - 3 * hyper_params_n))) for i in range(hyper_params_n)]
    hyper_params_nodes = [node if node > min_node_cnt else min_node_cnt for node in hyper_params_nodes]


def select_random_lambda():
    global hyper_params_lambdas
    n = random.random() * 6 - (5 - hyper_params_n)
    hyper_params_lambdas = 2 ** n


def get_features(df, y_label):
    if(which_feat == 1):
        features = hlp.DFToFeatureX(feature_model, df, x_label)
    elif(which_feat == 2):
        features = hlp.DFToFeatureX_W(feature_model, df, x_label)
    else:
        features = hlp.DFToFeatureX_W_Tdf(feature_model, idf_model, df, x_label)

    # features = hlp.PCA_reduce_feature(features, 50)

    features = hlp.standardize(features, True, False, d=2)
    # features = order_features(features, df.loc[:, [y_label]])
    return features


def order_features(features, y_out):
    features_pos, features_neg = hlp.splitPosAndNeg(features, y_out)
    column_list = hlp.analyseFeature(features_pos, features_neg, feature_dim)
    features = features[column_list]
    return features


def get_label(df, y_label):
    y_output = df.loc[:, [y_label]]
    y_output[y_output != 0] = 1
    y_output = y_output.round(0).astype(int)
    y_output['comp'] = 1 - y_output
    return y_output


def split_data(features, y_output):
    # x_train, y_train, x_test, y_test = hlp.shuffle_and_distribute(features, y_output, 20)
    x_train, y_train, x_test, y_test = hlp.splitData(features, y_output, 20)
    return x_train, y_train, x_test, y_test


def run_model(model, x_train, y_train, x_test, y_test, lambda_val):
    acc_eval = -1
    while acc_eval < 0:
        model_params = model.train_and_test_neural_network(x_train, y_train, x_test, y_test, lambda_val)
        acc_eval = model_params[2]
    return model_params


def run_classifier(x_train, y_train, x_test, y_test, y_label):
    results_for_y = pd.DataFrame(columns=result_column_list)
    for _ in range(num_samples):
        model = ff_n.FeedForward(hyper_params_nodes, hyper_params_learning_rt, epoch)
        feature_params_vals = [feature_name, feature_dim]
        model_params_vals = [model.getName(), model.getLayerCnt(), str(model.getNumNodes()),
                             hyper_params_lambdas, model.getEpoch(), str(model.getLearningRate())]
        print(model_params_vals)
        model_params = run_model(model, x_train, y_train, x_test, y_test, hyper_params_lambdas)

        res = pd.DataFrame([feature_params_vals + model_params_vals + model_params], columns=result_column_list)
        results_for_y = results_for_y.append(res, ignore_index=True)
        print(y_label + "==> " + str(model_params))
        print("====================================")
        select_random_nodes()
        select_random_lambda()
    return results_for_y


def train_for_each_label():
    for y_label in y_labels:
        df = hlp.getNotNull(df_whl, y_label)
        features = get_features(df, y_label)
        y_output = get_label(df, y_label)
        x_train, y_train, x_test, y_test = split_data(features, y_output)
        fin_result = run_classifier(x_train, y_train, x_test, y_test, y_label)

        file_name = y_label + "_" + feature_name + "_" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".csv"
        fin_result.to_csv(file_name, encoding='utf-8', index=False)


if not os.path.exists("Output1"):
    os.makedirs("Output1")
os.chdir("Output1")
train_for_each_label()
