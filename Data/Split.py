import pandas as pd
import numpy as np
import helper as hlp
import matplotlib.pyplot as plt


x_label = "Comments"
file = "Data/unique.csv"

df = hlp.readData(file, x_label)
y_labels = df.columns
y_labels = np.delete(y_labels, np.argwhere(y_labels == x_label))
for col in y_labels:
    train, test = hlp.stratified_sampling(df[[x_label, col]], df[col], 80)
    train.to_csv("Data/" + col + "_train.csv")
    test.to_csv("Data/" + col + "_test.csv")
