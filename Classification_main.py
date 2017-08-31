import helper as hlp
import pandas as pd
import idf
import ffmodel as ff
from datetime import datetime
import os
import convff
import rnn
import gensim
import math

data_csv_file = "TrimedData.csv"
y_labels = ["Praise", "Problem", "Solution", "Mitigation", "Neutrality", "Localization", "Summary"]
x_label = "Comments"

lambda_vals = [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20]

df_whl = hlp.readData(data_csv_file, x_label)

feature_model = gensim.models.Doc2Vec.load("comments2vec.d2v")
feature_model = gensim.models.Word2Vec.load("cmtWord2vec" + str(400) + ".d2v")
idf_model = idf.Idf.load("idf_model.json")
hyper_params = ["Model", "Prb_Label", "No of Layers", "No of Nodes", "lambda"]
output_attibute_list = ["Train Cost", "Test Cost", "Accuracy", "True Pos", "False Pos", "False Neg", "True Neg", "F1-Score"]
result_column_list = hyper_params + output_attibute_list


def get_features(df, y_label):
    # features = hlp.DFToFeatureX(feature_model, df, x_label)
    features = hlp.DFToFeatureX_W(feature_model, df, x_label)
    # features = hlp.DFToFeatureX_W_Tdf(feature_model, idf_model, df, x_label)
    # features = hlp.PCA_reduce_feature(features, 50)

    features = hlp.standardize(features, True, False, d=2)
    # features = order_features(features, df.loc[:, [y_label]])
    return features


def order_features(features, y_out):
    features_pos, features_neg = hlp.splitPosAndNeg(features, y_out)
    column_list = hlp.analyseFeature(features_pos, features_neg, 400)
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


def run_for_all_lambda(model, x_train, y_train, x_test, y_test, lambda_vals, y_label):
    results_for_y = pd.DataFrame(columns=result_column_list)
    model_params_vals = [model.getName(), y_label, model.getLayerCnt(), model.getNumNodes()]
    for lambda_val in lambda_vals:
        model_params = run_model(model, x_train, y_train, x_test, y_test, lambda_val)
        res = pd.DataFrame([model_params_vals + [lambda_val] + model_params], columns=result_column_list)
        results_for_y = results_for_y.append(res, ignore_index=True)
    return results_for_y


def run_for_all_labels():

    model = ff.FeedForward(200, 100)
    fin_result = pd.DataFrame(columns=result_column_list)

    for y_label in y_labels:
        df = hlp.getNotNull(df_whl, y_label)
        features = get_features(df, y_label)
        y_output = get_label(df, y_label)
        x_train, y_train, x_test, y_test = split_data(features, y_output)
        res_ys = run_for_all_lambda(model, x_train, y_train, x_test, y_test, lambda_vals, y_label)
        fin_result = fin_result.append(res_ys, ignore_index=True)

    if not os.path.exists("Output"):
        os.makedirs("Output")
    os.chdir("Output")
    file_name = model.getName() + str(model.getLayerCnt()) + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".csv"
    fin_result.to_csv(file_name, encoding='utf-8', index=False)


run_for_all_labels()
