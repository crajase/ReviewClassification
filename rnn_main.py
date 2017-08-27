import gensim
import idf
import matplotlib.pyplot as plt
import helper as hlp
import numpy as np
import data_clean as dc
import rnn_new
import pandas as pd


def DFToFeatureX_W_Tdf_for_rnn(feature_model, idf_model, df, x_label):
    # TODO: Check each step for correctness
    featuresX = []
    maxx = 1
    vec_len = feature_model.layer1_size
    df = df[x_label]
    for string in df:
        # TODO: Clean by removing stopwords
        words = dc.get_clean_wrd_list(string)
        word_idf = idf_model.calc_tf_idf(words)
        words_vec = []
        for word in words:
            if(word in feature_model):
                # feat = [ft * word_idf[word] for ft in feature_model[word]]
                feat = [ft for ft in feature_model[word]]
                words_vec.append(feat)
        if(len(words_vec) > 0):
            sent_vec = np.array(words_vec)
            featuresX.append(sent_vec)
        else:
            featuresX.append(np.zeros((1, vec_len)))

        if(len(words_vec) > maxx):
            maxx = len(words)
    featuresX = [np.pad(sent_vec1, ((0, maxx - sent_vec1.shape[0]), (0, 0)), mode='constant', constant_values=0) for sent_vec1 in featuresX]
    return np.array(featuresX)


def shuffle_and_distribute(features, y_output, testPerc):
    train_len = int((100 - testPerc) * features.shape[0] / 100)
    x_train = features[:train_len]
    x_test = features[train_len:]
    y_train = y_output[:train_len]
    y_test = y_output[train_len:]
    return x_train, y_train, x_test, y_test


def plotLineGraph(x_axis, y_axis, i, header):
    fig = plt.figure(i)
    fig.suptitle(header)
    plt.plot(x_axis, y_axis, color='green')


wrd_length = 100

data_csv_file = "TrimedData.csv"
y_label = "Problem"
x_label = "Comments"


df = hlp.readNonNullData(data_csv_file, x_label, y_label)

model_file_name = "cmtWord2vec" + str(wrd_length) + ".d2v"
feature_model = gensim.models.Word2Vec.load(model_file_name)
idf_model = idf.Idf.load("idf_model.json")

features = DFToFeatureX_W_Tdf_for_rnn(feature_model, idf_model, df, x_label)


y_output = df.loc[:, [y_label]]
if y_output.isnull().values.any():
    x = df[df.isnull()]
    print(x.shape)
    exit()
y_output[y_output != 0] = 1
y_output = y_output.round(0).astype(int)

x_train, y_train, x_test, y_test = shuffle_and_distribute(features, y_output, 20)

print("x-train = ", x_train.shape)
print("y-train = ", y_train.shape)
print("x-test = ", x_test.shape)
print("y-test = ", y_test.shape)
print("=========================================================")
model = rnn_new.Rnn(wrd_length)
prec = []
rec = []
f1 = []
acc = []
lambda_vals = [0]
for lambda_val in lambda_vals:
    acc_eval = -1
    print(lambda_val)
    while acc_eval < 0:
        print("cont")
        [train_cost, test_cost], acc_eval, conf_list, prf = model.train_and_test_neural_network(x_train, y_train, x_test, y_test, lambda_val)
    prec.append(prf[0])
    rec.append(prf[1])
    f1.append(prf[2])
    acc.append(acc_eval)
    print([train_cost, test_cost], acc_eval, conf_list, prf)

# plotLineGraph(lambda_vals, acc, 1, "Accuracy")
# plotLineGraph(lambda_vals, prec, 2, "Precision")
# plotLineGraph(lambda_vals, rec, 3, "Recall")
# plotLineGraph(lambda_vals, f1, 4, "F1-Score")


# plt.show()

print([train_cost, test_cost], acc_eval, conf_list, prf)
