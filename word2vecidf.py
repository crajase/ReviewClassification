import pandas as pd
import data_clean as dc
import idf
import os
import gensim

df = pd.read_csv("AllData.csv")
df.columns.values
headers = list(df.columns.values)

headers.remove("Comments")

df = df.drop(headers, axis=1)
df.head()

print(df.shape)
df.reset_index()
df = df[2000:]
print(df.shape)

comments = []
for index, row in df.iterrows():
    line = row["Comments"]
    comments.append(dc.get_clean_wrd_list(line))
i = 0

# Initialize model
size = 100
print("Building Word2vec model for the data starting from 2000")
model = gensim.models.Word2Vec(iter=10, size=size)
model.build_vocab(comments)
model.train(comments, total_examples=model.corpus_count, epochs=model.iter)
model_file_name = "cmtWord2vec" + str(size) + ".d2v"
model.save(model_file_name)
# load model
model = gensim.models.Word2Vec.load(model_file_name)
print(model)

print("================")
print("Building tf-idf model for the line starting from 2000")
idf_model = idf.Idf()
idf_model.build(comments)
idf_model.save("idf_model.json")
print(idf_model)
