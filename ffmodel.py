import tensorflow as tf
import math
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec


class FeedForward(object):
    def __init__(self, hl1_size, hl2_size):
        self.n_nodes_hl1 = hl1_size
        self.n_nodes_hl2 = hl2_size
        self.X = tf.placeholder('float', [None, None])
        self.Y = tf.placeholder('float')
        self.PredictedLabel = tf.placeholder('float')

    def getPrediction(self, data, featureSize, n_classes):
        hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([featureSize, self.n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))}

        l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
        # l1 = tf.nn.relu(l1)
        l1 = tf.nn.sigmoid(l1)

        l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
        # l2 = tf.nn.relu(l2)
        l2 = tf.nn.sigmoid(l2)

        output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

        reg = tf.reduce_sum(tf.square(hidden_layer_1['weights']))
        # reg += tf.reduce_sum(tf.square(hidden_layer_2['weights']))
        reg += tf.reduce_sum(tf.square(output_layer['weights']))

        return tf.nn.sigmoid(output), reg

    def compute_cost(self, prediction, Y, theta_sum, lambda_val):
        m = tf.reduce_sum(prediction) / tf.reduce_mean(prediction)
        cost_org = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
        # diff = tf.subtract(prediction, Y)
        # cost = tf.reduce_mean(tf.multiply(diff, diff))
        # singlecost = tf.multiply(Y, tf.log(prediction)) + tf.multiply((1 - Y), tf.log(1 - prediction))
        # cost_org = -1 * tf.reduce_mean(tf.reduce_sum(singlecost, axis=1))

        # cost = tf.reduce_mean(tf.exp(tf.abs(diff)))
        # regularization
        cost = cost_org + lambda_val * theta_sum / (2 * m)   # move square to individual elements
        return cost, cost_org

    def optimization(self, cost):
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
        return optimizer

    # The training and test data are passed as Pandas dataframe object
    def train_and_test_neural_network(self, train_x, train_y, test_x, test_y, lambda_val):
        featureSize = len(train_x.columns)
        n_classes = 2     # len(train_y.columns)
        prediction, theta_sum = self.getPrediction(self.X, featureSize, n_classes)
        cost, cost_org = self.compute_cost(prediction, self.Y, theta_sum, lambda_val)
        optimizer = self.optimization(cost)
        hm_epochs = 15000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_cost = 0
            train_cost_o = 0
            _, c, co = sess.run([optimizer, cost, cost_org], feed_dict={self.X: train_x, self.Y: train_y})
            if(math.isnan(c)):
                return [None, None], -1, None, None
            for epoch in range(hm_epochs):
                _, c, co = sess.run([optimizer, cost, cost_org], feed_dict={self.X: train_x, self.Y: train_y})
                train_cost = c
                train_cost_o = co
                if(epoch % 2500 == 0):
                    print("Iter ", epoch, " out of ", hm_epochs, " Cost: ", train_cost)

            predict, theta = sess.run([prediction, theta_sum], feed_dict={self.X: test_x})
            test_cost, test_cost_o = sess.run([cost, cost_org], feed_dict={prediction: predict, self.Y: test_y, theta_sum: theta})
            acc_eval, conf_list, prf = self.evaluatoinParameter(predict, sess, test_x, test_y)
            #print(predict[1:10])
            #print(test_y[1:10])
        return [[train_cost, train_cost_o], [test_cost, test_cost_o]], acc_eval, conf_list, prf

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

        pre, act = sess.run([predicted_label, actual_label], feed_dict={self.PredictedLabel: predict, self.Y: test_y})

        final_prediction = predicted_label[:, 0]
        original_act = actual_label[:, 0]
        # Confusion Matrix
        TP = tf.count_nonzero(final_prediction * original_act)
        TN = tf.count_nonzero((1 - final_prediction) * (1 - original_act))
        FP = tf.count_nonzero(final_prediction * (1 - original_act))
        FN = tf.count_nonzero((1 - final_prediction) * original_act)
        tp, tn, fp, fn = sess.run([TP, TN, FP, FN], feed_dict={self.PredictedLabel: predict, self.Y: test_y})

        # TODO: precision, recall, f-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return acc_eval, [[tp, fp], [fn, tn]], [precision, recall, f1]
