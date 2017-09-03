import helper as hlp
import idf
import ffmodel as ff
import ff_model_new as ff_g
import convff
import rnn
import gensim
import matplotlib.pyplot as plt
import math


def plotLineGraph(x_axis, y_axis, i, header):
    fig = plt.figure(i)
    fig.suptitle(header)
    plt.plot(x_axis, y_axis, color='green')


data_csv_file = "TrimedData.csv"
y_label = "Praise"
x_label = "Comments"
# lambda_vals = [4, 4.5, 5, 5.5, 6, 6.5] #[2, 2.25, 2.5, 2.75, 3, 3.5, 4]
# lambda_vals = [5, 10, 13, 15, 20, 30]
lambda_vals = [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15]
# lambda_vals = [10, 11, 12, 13, 14, 15]
lambda_vals = [2]
df = hlp.readData(data_csv_file, x_label)
df = hlp.getNotNull(df, y_label)
print(df.shape)
x = df.loc[:, [y_label]]
x.to_csv("check.csv")
# feature_model = gensim.models.Doc2Vec.load("comments2vec.d2v")
# features = hlp.DFToFeatureX(feature_model, df, x_label)

feature_model = gensim.models.Word2Vec.load("cmtWord2vec.d2v")
features = hlp.DFToFeatureX_W(feature_model, df, x_label)
# idf_model = idf.Idf.load("idf_model.json")
# features = hlp.DFToFeatureX_W_Tdf(feature_model, idf_model, df, x_label)
# exit()
# features = hlp.PCA_reduce_feature(features, 50)
y_output = df.loc[:, [y_label]]
y_output[y_output != 0] = 1
y_output = y_output.round(0).astype(int)
y_output['comp'] = 1 - y_output
features = hlp.standardize(features, True, False, d=2)

features_pos, features_neg = hlp.splitPosAndNeg(features, df.loc[:, [y_label]])
column_list = hlp.analyseFeature(features_pos, features_neg, 400)
features = features[column_list]

num_feature = len(features.columns)
print("num_feature = ", num_feature)
# x_train, y_train, x_test, y_test = hlp.shuffle_and_distribute(features, y_output, 20)
x_train, y_train, x_test, y_test = hlp.splitData(features, y_output, 20)

print("Check_num = ", len(x_train.columns))
print(y_test.shape)
print("=====================================================================================================")

print(y_label)
# model = ff.FeedForward(200, 100)
model = ff_g.FeedForward([200, 100, 50, 25])
# model = conv.Conv(100, 50)
# model = convff.ConvFF()
# model = rnn.Rnn(num_feature)
f1 = []
acc = []
for lambda_val in lambda_vals:
    acc_eval = -1
    print(lambda_val)
    while acc_eval < 0:
        print("cont")
        model_params = model.train_and_test_neural_network(x_train, y_train, x_test, y_test, lambda_val)
        acc_eval = model_params[2]

    acc.append(acc_eval)
    f1.append(model_params[7])
    print(model_params)

plotLineGraph(lambda_vals, acc, 1, "Accuracy")
# plotLineGraph(lambda_vals, prec, 2, "Precision")
# plotLineGraph(lambda_vals, rec, 3, "Recall")
plotLineGraph(lambda_vals, f1, 4, "F1-Score")

print(y_label)
# plt.show()
