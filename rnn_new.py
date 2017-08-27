import tensorflow as tf
import math
import numpy as np
import pandas as pd


class Rnn(object):
    def __init__(self, num_feature):
        self.state_size = int(num_feature / 10)
        self.n_classes = 1
        self.wrd = 0
        self.sent = 0
        self.n_states = math.ceil(num_feature / self.state_size)
        self.num_feature = num_feature
        self.tot_num_feats = self.state_size + self.num_feature
        self.X = tf.placeholder('float', [None, None, self.num_feature])
        self.Y = tf.placeholder('float')
        self.PredictedLabel = tf.placeholder('float')
        self.init_cell_st = tf.placeholder('float', [self.state_size])
        self.init_output_st = tf.placeholder('float', [self.state_size])

    def rnn_cell(self, X_wrd, prev_cell_st, prev_output_st, weights, biases):
        self.wrd += 1
        X_wrd = tf.reshape(X_wrd, shape=[1, -1])
        input_vec = tf.concat([X_wrd, prev_output_st], axis=1)
        forget_g = tf.sigmoid(tf.matmul(input_vec, weights['forget']) + biases['forget'])
        input_g = tf.sigmoid(tf.matmul(input_vec, weights['input']) + biases['input'])
        calc_cell_st = tf.tanh(tf.matmul(input_vec, weights['cell']) + biases['cell'])
        current_cell_st = calc_cell_st * input_g + prev_cell_st * forget_g
        output_g = tf.sigmoid(tf.matmul(input_vec, weights['output']) + biases['output'])
        current_output_st = output_g * tf.tanh(current_cell_st)
        return current_cell_st, current_output_st

    def rnn_out(self, X, num_wrds, weights, biases):
        # tf.unstack X. loop over X.
        # call runn for each. do whatever u want
        wrds_vec = tf.unstack(X, num=num_wrds, axis=0)
        cell_st = tf.zeros([1, self.state_size], dtype=tf.float32)
        out_st = tf.zeros([1, self.state_size], dtype=tf.float32)
        self.sent += 1
        self.wrd = 0
        for wrd_vec in wrds_vec:
            vec_sum = tf.reduce_sum(wrd_vec)

            def call_rnn_cell():
                return self.rnn_cell(wrd_vec, cell_st, out_st, weights, biases)

            def no_op():
                return cell_st, out_st

            cell_st, out_st = tf.cond(tf.equal(vec_sum, 0), no_op, call_rnn_cell)

        output = tf.sigmoid(tf.matmul(out_st, weights['fcl']) + biases['fcl'])
        output = tf.reshape(output, shape=[-1])
        return output

    def getPrediction(self, data_x, num_wrds):
        weights = {'forget': tf.Variable(tf.random_normal([self.tot_num_feats, self.state_size])),
                   'input': tf.Variable(tf.random_normal([self.tot_num_feats, self.state_size])),
                   'cell': tf.Variable(tf.random_normal([self.tot_num_feats, self.state_size])),
                   'output': tf.Variable(tf.random_normal([self.tot_num_feats, self.state_size])),
                   'fcl': tf.Variable(tf.random_normal([self.state_size, self.n_classes]))}

        biases = {'forget': tf.Variable(tf.random_normal([self.state_size])),
                  'input': tf.Variable(tf.random_normal([self.state_size])),
                  'cell': tf.Variable(tf.random_normal([self.state_size])),
                  'output': tf.Variable(tf.random_normal([self.state_size])),
                  'fcl': tf.Variable(tf.random_normal([self.n_classes]))}

        tensorArr = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
        sents_vec = tensorArr.unstack(data_x)
        outArr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        i = tf.constant(0)

        def body(i, outArr):
            outArr = outArr.write(i, self.rnn_out(sents_vec.read(i), num_wrds, weights, biases))
            return [i + 1, outArr]

        def cond(i, outArr):
            return i < sents_vec.size()

        i, out = tf.while_loop(cond, body, [i, outArr])

        reg = tf.reduce_sum(weights['fcl'])
        return out.stack(), reg

    def compute_cost(self, prediction, Y, theta_sum, lambda_val):
        m = tf.reduce_sum(prediction) / tf.reduce_mean(prediction)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
        # diff = tf.subtract(prediction, Y)
        # cost = tf.reduce_mean(tf.square(diff))
        singlecost = tf.multiply(Y, tf.log(prediction))
        singlecost = singlecost + tf.multiply((1 - Y), tf.log(1 - prediction))
        cost = -1 * tf.reduce_mean(tf.reduce_sum(singlecost, axis=1))

        # cost = tf.reduce_mean(tf.exp(tf.abs(diff)))
        # regularization
        cost += lambda_val * theta_sum / (2 * m)   # move square to individual elements
        return cost

    def optimization(self, cost):
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
        # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
        return optimizer

    # The training and test data are passed as Pandas dataframe object
    def train_and_test_neural_network(self, train_x, train_y, test_x, test_y, lambda_val):
        print("incomming")
        num_wrds = train_x.shape[1]

        self.n_classes = 1     # len(train_y.columns)
        prediction, theta_sum = self.getPrediction(self.X, num_wrds)
        cost = self.compute_cost(prediction, self.Y, theta_sum, lambda_val)
        optimizer = self.optimization(cost)

        hm_epochs = 25
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_cost = 0
            _, train_cost = sess.run([optimizer, cost], feed_dict={self.X: train_x, self.Y: train_y})
            if(math.isnan(train_cost)):
                return [None, None], -1, None, None
            for epoch in range(hm_epochs):
                _, train_cost, thet = sess.run([optimizer, cost, theta_sum], feed_dict={self.X: train_x, self.Y: train_y})
                print(train_cost, thet)
                if(epoch % 5 == 0):
                    print("Iter ", epoch, " out of ", hm_epochs, " Cost: ", train_cost)
            print(thet)
            predict, theta = sess.run([prediction, theta_sum], feed_dict={self.X: test_x})
            test_cost = sess.run(cost, feed_dict={prediction: predict, self.Y: test_y, theta_sum: theta})
            acc_eval, conf_list, prf = self.evaluatoinParameter(predict, sess, test_x, test_y)
            print(theta)
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
