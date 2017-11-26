import pandas as pd
import numpy as np
import helper as hlp
import matplotlib.pyplot as plt


x_label = "Comments"
inter_file = "Data/Intermediate.csv"

data_csv_file = "Data/AllData5.csv"
create_inter = 0
if(create_inter == 0):
    df = hlp.readData(data_csv_file, x_label)
    y_labels = df.columns
    y_labels = np.delete(y_labels, np.argwhere(y_labels == x_label))
    df = hlp.NotNullAllLabels(df, y_labels)
    df.to_csv(inter_file)
else:
    df = pd.read_csv(inter_file)
    y_labels = df.columns
df[y_labels].plot()
duplicates = df[df[x_label].duplicated(keep=False)]
print(duplicates)
duplicates.to_csv("Data/duplicates.csv")
'''
# remove nasty single digits..and
x = df[x_label].apply(len)
df = df[x > 2]
'''

# Non Duplicates

unique = df[~df[x_label].duplicated(keep=False)]
print(unique)
unique.to_csv("Data/unique.csv")


print("DataFrame = ", df.shape)
print("Duplicates = ", duplicates.shape)
print("Unique = ", unique.shape)
plt.show()
