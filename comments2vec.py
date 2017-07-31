import pandas as pd
import re         #remove and add import cleandata
import os
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random
import numpy as np
import tensorflow as tf
import multiprocessing

def clean(data):
    data = re.sub("[^a-zA-Z?!.@]", " ", str(data))
    word_list = []
    for word in data.strip().split():
        word_lower = word.lower()
        if word_lower not in stop_words:
            word_list.append(word_lower)
    return word_list
    
df = pd.read_csv("AllData.csv")
df.columns.values
headers = list(df.columns.values)

headers.remove("Comments")

df = df.drop(headers, axis=1)
df.head()

print(df.shape)
df.reset_index()
#df = df[2000:]
print(df.shape)

comments = []
for index, row in df.iterrows():
    line = row["Comments"]
    line = re.sub("[^a-zA-Z?!]"," ", str(line))  #remove it
    words = [w.lower() for w in line.strip().split() if len(w)>=3]        #change it to cleanData
    comments.append(words)
i = 0
labeled_comments = []
#comments = comments[0:len(comments) - 200]
for comment in comments:
    #print comment
    sentence = LabeledSentence(words=comment, tags=["COMMENT_"+str(i)])
    labeled_comments.append(sentence)
    i += 1
    
#more dimensions mean more trainig them, but more generalized
num_features = 400
# Minimum word count threshold.
min_word_count = 1
# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
# Context window length.
context_size = 10
# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-5

# Initialize model
model = Doc2Vec(min_count=min_word_count,
	window=context_size, 
	size=num_features,
	sample=downsampling,
	negative=20,
	workers=num_workers)
	
model.build_vocab(labeled_comments)

# Train the model
# This may take a bit to run #20 is better

for epoch in range(20):
    print("Training iteration %d" % (epoch))
    random.shuffle(labeled_comments)
    model.train(labeled_comments,total_examples=model.corpus_count,epochs=model.iter)
#save model
if not os.path.exists("trained"):
    os.makedirs("trained")
model.save("comments2vec.d2v")
#load model
model = Doc2Vec.load("comments2vec.d2v")
print(model)

#x_train = []
#for i in range(100):
#    x_train.append(model.docvecs["COMMENT_"+str(i)])

#x_test = []
#for i in range(100,110):
#    x_test.append(model.docvecs["COMMENT_"+str(i)])
    
#y_train = y[0:100]
#y_test = y[100:110]


