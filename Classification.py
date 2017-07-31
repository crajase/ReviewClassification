import helper as hlp
import ffmodel as ff
import convff
import rnn
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import math

def plotLineGraph(x_axis, y_axis, i, header):
    fig = plt.figure(i)
    fig.suptitle(header)
    plt.plot(x_axis, y_axis, color='green')


data_csv_file = "TrimedData.csv"
y_label = "Praise"
x_label = "Comments"
num_feature = 100
# lambda_vals = [4, 4.5, 5, 5.5, 6, 6.5] #[2, 2.25, 2.5, 2.75, 3, 3.5, 4]
# lambda_vals = [5, 10, 13, 15, 20, 30]
lambda_vals = [0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10]
df = hlp.readNonNullData(data_csv_file, x_label, y_label)
print(df[1:3])
feature_model = Doc2Vec.load("comments2vec.d2v")
features = hlp.DFToFeatureX(feature_model, df, x_label)
#features = hlp.PCA_reduce_feature(features, 50)
y_output = df.loc[:, [y_label]]
if y_output.isnull().values.any():
    x = df[df.isnull()]
    print(x.shape)
    exit()
y_output[y_output != 0] = 1
y_output = y_output.round(0).astype(int)
features = hlp.standardize(features, True, False, d=1)

features_pos, features_neg = hlp.splitPosAndNeg(features, df.loc[:, [y_label]])
column_list = hlp.analyseFeature(features_pos, features_neg, 400)
features = features[column_list]
# features = hlp.PCA_reduce_feature(features, 256)
num_feature = len(features.columns)
print("num_feature = ", num_feature)
x_train, y_train, x_test, y_test = hlp.shuffle_and_distribute(features, y_output, 20)
# x_train, y_train, x_test, y_test = hlp.splitData(features, y_output, 200)
print("Check_num = ", len(x_train.columns))
print("=====================================================================================================")
# model = ff.FeedForward(200, 100)
# model = conv.Conv(100, 50)
# model = convff.ConvFF()
model = rnn.Rnn(num_feature)
prec = []
rec = []
f1 = []
acc = []
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

plotLineGraph(lambda_vals, acc, 1, "Accuracy")
plotLineGraph(lambda_vals, prec, 2, "Precision")
plotLineGraph(lambda_vals, rec, 3, "Recall")
plotLineGraph(lambda_vals, f1, 4, "F1-Score")


plt.show()
