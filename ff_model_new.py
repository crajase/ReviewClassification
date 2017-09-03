import tensorflow as tf
import math
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec


class FeedForward(object):
    def __init__(self, num_nodes):
        self.n_nodes = num_nodes
        self.X = tf.placeholder('float', [None, None])
        self.Y = tf.placeholder('float')
        self.PredictedLabel = tf.placeholder('float')

    def getPrediction(self, data, featureSize, n_classes):
        weights_hidden = {}
        biases_hidden = {}
        dim_l = featureSize
        for idx, layer_sz in enumerate(self.n_nodes):
            layer_idx = idx + 1
            weights_hidden[layer_idx] = tf.Variable(tf.random_normal([dim_l, layer_sz]) * 0.01)
            biases_hidden[layer_idx] = tf.Variable(tf.random_normal([layer_sz]) * 0.01)
            dim_l = layer_sz

        output_layer = {'weights': tf.Variable(tf.random_normal([dim_l, n_classes]) * 0.01),
                        'biases': tf.Variable(tf.random_normal([n_classes]) * 0.01)}

        last_act = data
        for idx, layer_sz in enumerate(self.n_nodes):
            layer_idx = idx + 1
            z = tf.add(tf.matmul(last_act, weights_hidden[layer_idx]), biases_hidden[layer_idx])
            # last_act = tf.nn.relu(z)
            last_act = tf.nn.tanh(z)

        output = tf.matmul(last_act, output_layer['weights']) + output_layer['biases']

        reg = 0

        for idx, layer_sz in enumerate(self.n_nodes):
            layer_idx = idx + 1
            reg += tf.reduce_sum(tf.square(weights_hidden[layer_idx]))
        reg += tf.reduce_sum(tf.square(output_layer['weights']))

        return tf.nn.sigmoid(output), reg

    def compute_cost(self, prediction, Y, theta_sum, lambda_val):
        m = tf.reduce_sum(prediction) / tf.reduce_mean(prediction)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
        # diff = tf.subtract(prediction, Y)
        # loss = tf.reduce_mean(tf.multiply(diff, diff))
        # loss_diff = tf.multiply(Y, tf.log(prediction)) + tf.multiply((1 - Y), tf.log(1 - prediction))
        # loss = -1 * tf.reduce_mean(tf.reduce_sum(loss_diff, axis=1))

        # cost = tf.reduce_mean(tf.exp(tf.abs(diff)))
        # regularization
        cost = loss + lambda_val * theta_sum / (2 * m)   # move square to individual elements
        return cost, loss

    def optimization(self, cost):
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
        return optimizer

    # The training and test data are passed as Pandas dataframe object
    def train_and_test_neural_network(self, train_x, train_y, test_x, test_y, lambda_val):
        featureSize = len(train_x.columns)
        n_classes = len(train_y.columns)
        prediction, theta_sum = self.getPrediction(self.X, featureSize, n_classes)
        cost, loss = self.compute_cost(prediction, self.Y, theta_sum, lambda_val)
        optimizer = self.optimization(cost)
        hm_epochs = 5000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_cost = 0
            train_loss = 0
            _, c, co = sess.run([optimizer, cost, loss], feed_dict={self.X: train_x, self.Y: train_y})
            if(math.isnan(c)):
                return [None, None, -1, None, None, None, None, None]
            for epoch in range(hm_epochs):
                _, c, los = sess.run([optimizer, cost, loss], feed_dict={self.X: train_x, self.Y: train_y})
                train_cost = c
                train_loss = los
                if(epoch % int(hm_epochs / 4) == 0):
                    print("Iter ", epoch, " out of ", hm_epochs, " Cost: ", train_cost)

            print("Iter ", epoch, " out of ", hm_epochs, " Cost: ", train_cost)
            predict, theta = sess.run([prediction, theta_sum], feed_dict={self.X: test_x})
            print(predict[1:5])
            # print(test_y)
            test_cost, test_loss = sess.run([cost, loss], feed_dict={prediction: predict, self.Y: test_y, theta_sum: theta})
            eval_params = self.evaluatoinParameter(predict, sess, test_x, test_y)
        return [train_loss, test_loss] + eval_params

    def evaluatoinParameter(self, predict, sess, test_x, test_y):
        n_classes = len(test_y.columns)
        if(n_classes == 1):
            predicted_label = tf.round(self.PredictedLabel)
            actual_label = self.Y
        else:
            predicted_label = 1 - tf.argmax(self.PredictedLabel, 1)
            actual_label = 1 - tf.argmax(self.Y, 1)

        # Accuracy
        correct = tf.equal(predicted_label, actual_label)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc_eval = accuracy.eval({self.PredictedLabel: predict, self.Y: test_y})

        pre, act = sess.run([predicted_label, actual_label], feed_dict={self.PredictedLabel: predict, self.Y: test_y})

        # Confusion Matrix
        TP = tf.count_nonzero(predicted_label * actual_label)
        TN = tf.count_nonzero((1 - predicted_label) * (1 - actual_label))
        FP = tf.count_nonzero(predicted_label * (1 - actual_label))
        FN = tf.count_nonzero((1 - predicted_label) * actual_label)
        tp, tn, fp, fn = sess.run([TP, TN, FP, FN], feed_dict={self.PredictedLabel: predict, self.Y: test_y})

        # TODO: precision, recall, f-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return [acc_eval, tp, fp, fn, tn, f1]

    def getName(self):
        return "Feed_Forward"

    def getLayerCnt(self):
        return len(self.n_nodes)

    def getNumNodes(self):
        return str(self.n_nodes_hl1) + ", " + str(self.n_nodes_hl2)
