# Copyright 2017 Chengkai Wu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from dataset import Dataset
import numpy as np
import tensorflow as tf


class Processor:
    """
    This class returns a multi-label processor with build-in algorithms of mlknn,... etc.

    For example:

    ```python
    proc = ml.Processor(train, test)
    pred = proc.mlknn_trainer(k=2, mode='euclidean')
    y, y_ = model.predictor(pred)
    ```

    """

    def __init__(self, train: Dataset, test: Dataset):

        self.train = train
        self.test = test
        self.x_dim = len(train.xs[0])
        self.y_dim = len(train.ys[0])
        self.y = None
        self.y_ = None
        self.__pred = None
        self.__init_placeholder()
        self.__init_session()

    def __init_placeholder(self):
        """
        initialize placeholders for all tf based algorithms, including
        tr_x, tr_y, te_x, te_y
        :return: None
        """
        x_dim = self.x_dim
        y_dim = self.y_dim

        with tf.variable_scope('common'):
            self.__placeholder = (
                tf.placeholder(dtype=tf.float32, shape=[None, x_dim]),
                tf.placeholder(dtype=tf.int32, shape=[None, y_dim]),
                tf.placeholder(dtype=tf.float32, shape=[x_dim]),
                tf.placeholder(dtype=tf.int32, shape=[y_dim])
            )

    def __init_session(self):
        """
        initialize a tf session for this object
        :return: None
        """
        self.__sess = tf.Session()

    @ staticmethod
    def __eval_hamming_loss(y, y_):

        return np.sum(np.not_equal(y, y_))/np.size(y)

    @ staticmethod
    def __eval_accuracy(y, y_):

        y_and = np.sum(np.logical_and(y, y_), axis=1)
        y_or = np.sum(np.logical_or(y, y_), axis=1)
        # to avoid y_or == 0
        y_and = y_and + np.equal(y_or, 0)
        y_or = y_or + np.equal(y_or, 0)

        return np.mean(y_and/y_or)

    @ staticmethod
    def __eval_exact_match(y, y_):

        return sum(list(map(lambda a, b: a == b, y, y_)))/len(y)

    @ staticmethod
    def __eval_f1(y, y_):

        count_y = np.sum(y, axis=1) + (np.sum(y, axis=1) == 0)
        count_y_ = np.sum(y_, axis=1) + (np.sum(y_, axis=1) == 0)
        y_and = np.sum(np.logical_and(y, y_), axis=1)
        p = y_and/count_y_
        r = y_and/count_y
        f1 = np.mean(2*p*r/((p+r)+np.equal(p+r, 0)))

        return f1

    @ staticmethod
    def __eval_macro_f1(y, y_):

        count_y = np.sum(y, axis=0) + (np.sum(y, axis=0) == 0)
        count_y_ = np.sum(y_, axis=0) + (np.sum(y_, axis=0) == 0)
        y_and = np.sum(np.logical_and(y, y_), axis=0)
        p = y_and/count_y_
        r = y_and/count_y
        macro_f1 = np.mean(2*p*r/((p+r)+np.equal(p+r, 0)))

        return macro_f1

    @ staticmethod
    def __eval_micro_f1(y, y_):

        return 2*np.sum(np.logical_and(y, y_))/(np.sum(y) + np.sum(y_))

    def predictor(self, pred=None, test: Dataset=None):
        """
        To predict unlabeled data (or testing data) with a predictor pred
        :param pred: a predictor that has been trained in object
        :param test: the optional test dataset to use in predictor
        :return: gold_standard labels y and predicted labels y_
        """
        # if no pred parameter is filled, use a most recent used predictor
        if pred is None:
            pred = self.__pred

        # get global placeholders and sessions
        tr_x, tr_y, te_x, te_y = self.__placeholder
        sess = self.__sess

        # y is the gold_standard output and y_ is the predicted output
        y = []
        y_ = []
        inner_tr_x = self.train.xs
        inner_tr_y = self.train.ys
        # if there is no new test data input, use the global test data. if there is, use the input data.
        if test is None:
            test = self.test
        # run predicting procedures
        for i in range(test.num):
            # select data in each term
            inner_te_x = test.xs[i]
            inner_te_y = test.ys[i]
            # predicting
            feed_dict = {tr_x: inner_tr_x, te_x: inner_te_x, tr_y: inner_tr_y, te_y: inner_te_y}
            y_pred = sess.run(pred, feed_dict=feed_dict)
            # store y and y_ into lists
            y.append([int(ind) for ind in inner_te_y])
            y_.append(list(y_pred))

        self.y = y
        self.y_ = y_

        return y, y_

    def evaluator(self, y=None, y_=None):

        # only when y and y_ are both not None, the real input will be used.
        if y is None or y_ is None:
            y, y_ = self.y, = self.y_

        return {
            'hamming_loss': self.__eval_hamming_loss(y, y_),
            'accuracy': self.__eval_accuracy(y, y_),
            'exact_match': self.__eval_exact_match(y, y_),
            'f1': self.__eval_f1(y, y_),
            'macro_f1': self.__eval_macro_f1(y, y_),
            'micro_f1': self.__eval_micro_f1(y, y_),
        }

    def mlknn_trainer(self, k=10, mode='euclidean'):
        """
        to train predicting parameters with MLkNN.
        The algorithm is based on article
        'ML-KNN: A lazy learning approach to multi-label learning', Min-Liang Zhang, Zhi-Hua Zhou, 2006

        :param k: The number of nearest nodes in choosing knn
        :param mode: The distance mode, including 'euclidean' and 'manhattan'
        :return: pred node in tf, used in function o.predictor
        """

        # this line helps to limit current scope of tf variables. when initialize
        # variables, use
        #   tf.variables_initializer(tf.global_variables('mlknn'))
        # instead of
        #   tf.global_variables_initializer()

        with tf.variable_scope('mlknn'):

            # to get the dimension value of x-vectors and y-vectors
            y_dim = self.y_dim

            # constants

            label_indexes = tf.constant(list(range(y_dim)), dtype=tf.int32)

            # placeholders for training and testing data in tensorflow
            tr_x, tr_y, te_x, te_y = self.__placeholder

            # variables for model

            # c1_map[y_dim,k] represents the number of occasions when "a node has label l,
            # and its knn exactly contains c nodes with label l."
            # c0_map is its opposite, that suggests the happened occasion number of "a node
            # without label l has c nodes with label l"
            # e.g. if there are exactly 2 nodes (among the total k nodes) around a sample
            # node that has a same label l as sample does, then c1_map[l,2] += 1
            c1_map = tf.Variable(tf.zeros([y_dim, k+1], dtype=tf.float32))
            c0_map = tf.Variable(tf.zeros([y_dim, k+1], dtype=tf.float32))
            p1_map = tf.Variable(tf.zeros([y_dim, k+1], dtype=tf.float32))
            p0_map = tf.Variable(tf.zeros([y_dim, k+1], dtype=tf.float32))

            # choose the distance mode in ['euclidean', 'manhattan'] and
            # calculate the distance between training xs and a test sample
            distance = tf.constant(0)
            if mode not in ['euclidean', 'manhattan']:
                raise Exception('illegal distance mode')
            if mode == 'euclidean':
                distance = tf.reduce_sum(tf.squared_difference(tr_x, [te_x]), axis=1)
            if mode == 'manhattan':
                distance = tf.reduce_sum(tf.abs(tr_x - [te_x]), axis=1)

            # STEP1: kNN
            # find the k-nearest-nodes of testing sample te_x

            # 1. This line captures the k minimum distance values and their indexes.
            # The negative function is used to turn a minimization problem into a cor-
            # responding maximization problem
            values, knn_indexes = tf.nn.top_k(tf.negative(distance), k=k, sorted=False)

            # STEP2: Count Labels' Number in te_x's kNN
            # c is a y_dim-dimensional vector that c_i is the number of  knn training
            # data points that have the label y_i, around a specific testing data.

            # 1. reshape indexes to match the usage of gather Op
            knn_indexes = tf.reshape(knn_indexes, [k, 1])
            # 2. gather the label lists of kNN nodes and sum them in each label dimension
            c = tf.reduce_sum(tf.gather_nd(tr_y, knn_indexes), axis=0)

            # STEP3: Accumulate C-Maps

            # 1. pair label indexes and c to denote a position in c-maps
            pair = tf.stack([label_indexes, c], axis=1)
            # 2. at each position of c1_map, add 1 if te_y == 1, else add 0
            c1_add = tf.scatter_nd_add(c1_map, pair, tf.to_float(te_y))
            # 3. at each position of c0_map, add 1 if te)y == 0, else add 0
            c0_add = tf.scatter_nd_add(c0_map, pair, 1-tf.to_float(te_y))
            # 4. to create a tf node that run add Ops together
            c_map_add = (c1_add, c0_add)

            # STEP4: Derive Value of P(Hb)*P(Ec|Hb)   b={0,1}
            # the argmax of this value suggests the better hypothesis of whether sample
            # have a label or not. Here we calculate P(Hb) and P(Ec|Hb) respectively

            # 1. s = 1 for Laplace smoothing, label_count is a vector. each component
            # label_count[l] stores the number of data points that have label l.
            s = 1
            label_count = tf.matmul([tf.reduce_sum(c1_map, axis=1)], tf.ones([1, k+1]), transpose_a=True)
            # 2.1. calculate the estimation of prior probability P(H1), which is the
            # probability of labels to be used. The estimation is given with the
            # ratio for labels to be used in the training set.
            p_h1 = tf.divide(s + label_count, 2 * s + self.train.num)
            # 2.2. P(H0) = 1 - P(H1)
            p_h0 = 1 - p_h1
            # 3 calculate the estimation of  posterior probability P(Ec|Hb), which suggests
            # "knowing one node having (or not having) a certain label l, the possibility that
            # there are exactly c nodes having this label l in its knn"
            p_ec_h1 = tf.divide((s + c1_map), (s * (k + 1) + label_count))
            p_ec_h0 = tf.divide((s + c0_map), (s * (k + 1) + self.train.num - label_count))
            # 4. store the result Pb = P(Hb)*P(Ec|Hb)   b={0,1} in p1 and p2, both are Variables
            p1_update = tf.assign(p1_map, p_ec_h1*p_h1)
            p0_update = tf.assign(p0_map, p_ec_h0*p_h0)
            p_map_update = (p1_update, p0_update)

            # STEP 5: ARGMAX
            # find the category vector y_ which is argmax(p) in {0, 1}
            te_p = tf.stack([tf.gather_nd(p0_map, pair), tf.gather_nd(p1_map, pair)])
            pred = tf.argmax(te_p)

        # PROCESS
        # initialize all variable (parameters) in scope of 'mlknn' algorithm
        sess = self.__sess
        sess.run(tf.variables_initializer(tf.global_variables('mlknn')))

        # run training procedures
        for i in range(self.train.num):
            # when using training set, data is divided into sample p and set(X-p) to refresh c-maps
            print('train the %d-th data within the total of %d data' % (i, self.train.num))
            inner_tr_x = list(self.train.xs)
            inner_te_x = inner_tr_x.pop(i)
            inner_tr_y = list(self.train.ys)
            inner_te_y = inner_tr_y.pop(i)
            # training
            feed_dict = {tr_x: inner_tr_x, te_x: inner_te_x, tr_y: inner_tr_y, te_y: inner_te_y}
            sess.run(c_map_add, feed_dict=feed_dict)
        # after going through all training points, update p_map once
        sess.run(p_map_update)

        self.__pred = pred
        return pred


