# Copyright 2017 Chengkai Wu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import dataset as ds
import multilabel as ml
import os
import numpy as np

dataset = ds.Dataset('CAL500')
fold_iter = dataset.divide()

for fold in fold_iter:
    train = fold['train']
    test = fold['test']
    model = ml.Processor(train, test)
    # pred = model.mlknn_trainer(k=10)
    pred = model.bpmll_trainer(m=500)
    y, y_ = model.predictor(pred)
    print(model.evaluator(y, y_))

# train = ds.Dataset()
# test = ds.Dataset()
# train.create_dataset([[1, 1], [2, 2], [13, 3], [0, 8], [15, 0]],
#                      [[1, 0], [0, 1], [1, 0], [1, 1], [0, 0]])
# test.create_dataset([[3, 2], [14, 5], [1, 13]],
#                     [[1, 1], [0, 0], [1, 0]])
# model = ml.Processor(train, test)
# pred = model.mlknn_trainer(k=2, mode='euclidean')
# pred = model.bpmll_trainer(m=10)
# y, y_ = model.predictor(pred)
# print(model.evaluator(y, y_))
