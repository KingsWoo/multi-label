# Copyright 2017 Chengkai Wu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from xml.etree import ElementTree as Et
import arff
import os
import random


class Dataset:

    xs = None
    ys = None
    x_names = None
    y_names = None
    num = None

    def __init__(self, name=None):
        """
        it is only allowed to get an exist dataset while creating the Dataset object,
        if wants to use own data, please use create_dataset method afterwards
        :param name: the name of the dataset if wants to extract it
        """
        if name is not None:
            self.get_exist_dataset(name)

    def get_exist_dataset(self, name):
        """
        fill the Dataset object with a exist dataset
        :param name: the name of the dataset
        :return:
        """
        if self.__if_dataset_exist(name) is False:
            raise Exception('cannot get dataset %s for it does not exist' % name)
        else:
            # read xml file
            path = os.getcwd() + '\\datasets\\%s\\%s.xml ' % (name, name)
            root = Et.ElementTree(file=path).getroot()

            # read arff file
            path = os.getcwd() + '\\datasets\\%s\\%s.arff' % (name, name)
            data = arff.load(open(path))

            # get y_names
            y_names = [child.attrib['name'] for child in root]

            # get x_names
            attr = [each[0] for each in data[u'attributes']]
            x_names = list(filter(lambda x: x not in y_names, attr))

            # get xs & ys
            x_len = len(x_names)
            y_len = len(y_names)
            xs = [each[0:x_len] for each in data[u'data']]
            ys = [each[x_len:x_len + y_len] for each in data[u'data']]

            # get sample number
            num = len(xs)

            # storage
            self.xs = xs
            self.ys = ys
            self.x_names = x_names
            self.y_names = y_names
            self.num = num

    def create_dataset(self, xs, ys, x_names=None, y_names=None):

        self.xs = xs
        self.ys = ys
        self.x_names = x_names
        self.y_names = y_names
        self.num = len(xs)
        self.__check_exception()

    def divide(self, test_rate=0.2, foldnum=10, mode='single'):

        """
        randomly divide dataset into training sets and testing sets, with single test/fold test modes
        :param test_rate: the ratio of data used in testing sets (only in single test mode)
        :param foldnum: number of folds (only in fold test mode)
        :param mode: choose mode in ('single','fold')
        :return: an iterator of (train_i, test_i)
        """

        tr_size = None
        te_size = None
        num = self.num

        if mode not in ('single', 'fold'):
            raise Exception('illegal mode option')

        if mode == 'single':
            te_size = round(num * test_rate)
            tr_size = num - te_size
            foldnum = 1

        if mode == 'fold':
            te_size = num // foldnum
            tr_size = te_size * foldnum - te_size
            foldnum = foldnum

        # randomly select and divide data into folds
        test = Dataset()
        train = Dataset()
        used_index = random.sample(range(num), tr_size+te_size)
        for i in range(foldnum):
            te_index = used_index[i*te_size:(i+1)*te_size]
            tr_index = list(filter(lambda x: x not in te_index, used_index))
            test.create_dataset(xs=[self.xs[each] for each in te_index],
                                ys=[self.ys[each] for each in te_index],
                                x_names=self.x_names,
                                y_names=self.y_names)
            train.create_dataset(xs=[self.xs[each] for each in tr_index],
                                 ys=[self.ys[each] for each in tr_index],
                                 x_names=self.x_names,
                                 y_names=self.y_names)
            yield {'test': test, 'train': train}

    def __check_exception(self):

        xs = self.xs
        ys = self.ys
        x_names = self.x_names
        y_names = self.y_names

        if len(xs) != len(ys):
            raise Exception('length of xs and ys are not equal')
        if xs is not None and x_names is not None and len(x_names) != len(xs[0]):
            raise Exception('vector dimension of x does not match its attributes')
        if ys is not None and y_names is not None and len(y_names) != len(ys[0]):
            raise Exception('vector dimension of y does not match its attributes')

    @ staticmethod
    def __if_dataset_exist(name):
        """
        To check if a dataset exists
        :return: True for existence and False for not
        """
        dataset_list = os.listdir(os.getcwd() + '//datasets')
        return name in dataset_list
