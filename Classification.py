import helper as hlp
import ffmodel as ff
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt


def plotLineGraph(x_axis, y_axis, i, header):
    fig = plt.figure(i)
    fig.suptitle(header)
    plt.plot(x_axis, y_axis, color='green')


data_csv_file = "TrimedData.csv"
y_label = "Praise"
x_label = "Comments"
lambda_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
df = hlp.readNonNullData(data_csv_file)
feature_model = Doc2Vec.load("comments2vec.d2v")
features = hlp.DFToFeatureX(feature_model, df, x_label)
y_output = df.loc[:, [y_label]]
y_output[y_output != 0] = 1
y_output = y_output.round(0).astype(int)
features = hlp.standardize(features, d=2)
x_train, y_train, x_test, y_test = hlp.splitData(features, y_output, 200)

print("==========================")
ffmodel = ff.FeedForward(100, 100)
prec = []
rec = []
f1 = []
acc = []
for lambda_val in lambda_vals:
    [train_cost, test_cost], acc_eval, conf_list, prf = ffmodel.train_and_test_neural_network(x_train, y_train, x_test, y_test, lambda_val)
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
