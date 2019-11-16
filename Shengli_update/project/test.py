#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-30 下午11:08
# @Author  : zejin
# @File    : test.py
from data_prepare.point_to_label.data_util_shallow import *
from models.Shallowmodel.training.svm_train import presentation,evaluate_plot
if __name__ == "__main__":
    train_data, validation_data, test_data = get_input()
    samples = test_data[2]
    # for keys in samples.keys():
    #     print(len(samples[keys][0]))
    #evaluate_plot()
    presentation(samples)