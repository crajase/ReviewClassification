import tensorflow as tf
import math
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec


class ConvFF(object):
    def __init__(self):
        self.X = tf.placeholder('float', [None, None])
        self.Y = tf.placeholder('float')
        self.PredictedLabel = tf.placeholder('float')
        self.filter_size = 5

    # dimension of W is filter_hgt x number_of_act_unit
    # dimension of x is num_samples x num_feature
    def conv2d(self, x, W, num_act_unit):
        x = tf.reshape(x, shape=[-1, num_act_unit, self.filter_size])
        x = tf.transpose(x, [0, 2, 1])
        # resultant x dimension is num_samples x num_filter_hgt x number_of act_unit
        mul = x * W
        return tf.reduce_sum(mul, axis=1)

    def getPrediction(self, x, featureSize, n_classes):
        lr1 = int(featureSize / self.filter_size)
        lr2 = int(lr1 / self.filter_size)
        weights = {'W_conv1': tf.Variable(tf.random_normal([self.filter_size, lr1])),
                   'W_conv2': tf.Variable(tf.random_normal([self.filter_size, lr2])),
                   'out': tf.Variable(tf.random_normal([lr2, n_classes]))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([lr1])),
                  'b_conv2': tf.Variable(tf.random_normal([lr2])),
                  'out': tf.Variable(tf.random_normal([n_classes]))}
        partW1 = [(i + 0) / lr1 for i in range(lr1)]
        partW2 = [(i + 0) / lr2 for i in range(lr2)]

        conv1 = tf.nn.tanh(self.conv2d(x, weights['W_conv1'], lr1) + biases['b_conv1'])

        conv2 = tf.nn.tanh(self.conv2d(conv1, weights['W_conv2'], lr2) + biases['b_conv2'])

        output = tf.nn.sigmoid(tf.matmul(conv2, weights['out']) + biases['out'])

        reg = tf.reduce_sum(tf.reduce_sum(tf.square(weights['W_conv1']), axis=0) * partW1)
        reg += tf.reduce_sum(tf.reduce_sum(tf.square(weights['W_conv2']), axis=0) * partW2)
        reg += tf.reduce_sum(tf.square(weights['out']))

        return output, reg

    def compute_cost(self, prediction, Y, theta_sum, lambda_val):
        m = tf.reduce_sum(prediction) / tf.reduce_mean(prediction)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
        # diff = tf.subtract(prediction, Y)
        # cost = tf.reduce_mean(tf.square(diff))
        singlecost = tf.multiply(Y, tf.log(prediction)) + tf.multiply((1 - Y), tf.log(1 - prediction))
        cost = -1 * tf.reduce_mean(tf.reduce_sum(singlecost, axis=1))

        # cost = tf.reduce_mean(tf.exp(tf.abs(diff)))
        # regularization
        cost += lambda_val * theta_sum / (2 * m)   # move square to individual elements
        return cost

    def optimization(self, cost):
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
        return optimizer

    # The training and test data are passed as Pandas dataframe object
    def train_and_test_neural_network(self, train_x, train_y, test_x, test_y, lambda_val):
        featureSize = len(train_x.columns)
        n_classes = 1     # len(train_y.columns)
        prediction, theta_sum = self.getPrediction(self.X, featureSize, n_classes)
        cost = self.compute_cost(prediction, self.Y, theta_sum, lambda_val)
        optimizer = self.optimization(cost)
        hm_epochs = 10000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_cost = 0
            _, c = sess.run([optimizer, cost], feed_dict={self.X: train_x, self.Y: train_y})
            if(math.isnan(c)):
                return [None, None], -1, None, None
            for epoch in range(hm_epochs):
                _, c = sess.run([optimizer, cost], feed_dict={self.X: train_x, self.Y: train_y})
                train_cost = c
                if(epoch % 2500 == 0):
                    print("Iter ", epoch, " out of ", hm_epochs, " Cost: ", train_cost)

            predict, theta = sess.run([prediction, theta_sum], feed_dict={self.X: test_x})
            test_cost = sess.run(cost, feed_dict={prediction: predict, self.Y: test_y, theta_sum: theta})
            acc_eval, conf_list, prf = self.evaluatoinParameter(predict, sess, test_x, test_y)
            print(predict[1:10])
            print(test_y[1:10])
        return [train_cost, test_cost], acc_eval, conf_list, prf

    def evaluatoinParameter(self, predict, sess, test_x, test_y):
        n_classes = 1       # len(test_y.columns)
        if(n_classes == 1):
            predicted_label = tf.round(self.PredictedLabel)
            actual_label = self.Y
        else:
            predicted_label = tf.argmax(self.PredictedLabel, 1)
            actual_label = tf.argmax(self.Y, 1)

        # Accuracy
        correct = tf.equal(predicted_label, actual_label)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc_eval = accuracy.eval({self.PredictedLabel: predict, self.Y: test_y})

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
        return acc_eval, [[tp, fp], [fn, tn]], [precision, recall, f1]
