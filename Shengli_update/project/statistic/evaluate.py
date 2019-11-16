#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-15 下午1:42
# @Author  : zejin
# @File    : evaluate.py

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from statistic.analysis import get_labels
def evaluate(true_labels,predict_labels,seq_length, test = False,specific_thres = 0.5):
    """

    :param true_labels:
    :param predict_labels:
    :return:
    """
    Precision = 0; Precision_list = []
    Recall = 0   ; Recall_list = []
    F1 = 0       ; F1_list = []
    FPR = 0      ; FPR_list = []
    for thres in range(0,101):
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        thres = thres * 0.01
        if not test:
            thres = specific_thres
        for i,Cur_len in enumerate(seq_length):
            for tlabel,plabel in zip(true_labels[i][:Cur_len],predict_labels[i][:Cur_len]):
                if tlabel[1] == 1 and plabel[1] >= thres:
                    TP += 1
                elif tlabel[1] == 1 and plabel[1] < thres:
                    FN += 1
                elif tlabel[1] == 0 and plabel[1] >= thres:
                    FP += 1
                else:
                    TN += 1
        if TP + FP == 0:
            Precision = 0
        else:
            Precision = TP/(TP+FP)
        if TP + FN == 0:
            Recall = 0
        else:
            Recall = TP/(TP+FN)
        if Precision + Recall == 0:
            F1 = 0
        else:
            F1 = 2*Precision*Recall/(Precision+Recall)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP/(FP+TN)
        Precision_list.append(Precision)
        Recall_list.append(Recall)
        F1_list.append(F1)
        FPR_list.append(FPR)
        if not test:
            break
    if not test:
        return Precision,Recall,F1,[sum([TP,TN,FP,FN]),sum(seq_length)]
    else:
        return Precision_list, Recall_list, F1_list ,FPR_list
def get_threshold_curve(sample_len,threshold_map):
    """
    根据样本的长度sample_len 得到相应的阈值线, 阈值线的长度为sample_len
    :param sample_len:
    :param threshold_map:
    :return:
    """
    corr_key = 0
    for key in sorted(threshold_map.keys()):
        if key-sample_len<0:
            continue
        else:
            corr_key = key
            break
    if corr_key == 0:
        corr_key = sorted(threshold_map.keys())[-1]
    x_values,y_values = threshold_map[corr_key][0],threshold_map[corr_key][1]
    threshold_curve = []
    for i,x in enumerate(x_values[:-1]):
        k = (y_values[i]-y_values[i+1])/(x_values[i]-x_values[i+1])
        b = y_values[i] - k*x_values[i]
        for c_x in range(x_values[i],x_values[i+1]):
            threshold_curve.append(k*c_x+b)
    threshold_curve.append(threshold_curve[-1])
    #plt.plot(threshold_curve)
    #plt.show()
    return threshold_curve
def evaluate_sample(t_labels,p_labels,threshold_curve):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    for i,p_label in enumerate(p_labels):
        if t_labels[i] == 1 and p_label >= threshold_curve[i]:
            TP += 1
        elif t_labels[i] == 1 and p_label < threshold_curve[i]:
            FN += 1
        elif t_labels[i] == 0 and p_label <= threshold_curve[i]:
            TN += 1
        else:
            FP += 1
    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    if TP + FN == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = 2 * Precision * Recall / (Precision + Recall)
    if FP + TN == 0:
        FPR = 0
    else:
        FPR = FP / (FP + TN)
    return [Precision, Recall, F1,FPR], [TP, TN, FP, FN]
def evaluate_using_threshold_curve(count = 3):
    with open('./threshold_curve/threshold_curve_28_count_'+str(count)+'.pkl','rb') as file:
        threshold_map = pickle.load(file)   # key 为储层深度，value为[x_list, y_list]
    print(threshold_map.keys())
    # 首先使用 trace_range = 1的数据进行验证
    count = 1
    true_labels_all,predict_labels_all = get_labels(count=count)
    TP_all = 0; TN_all = 0; FP_all = 0; FN_all = 0
    precision_all = []; recall_all = []; f1_all = []
    for sample_no,predict_labels in enumerate(predict_labels_all):
        sample_len = len(predict_labels)
        threshold_curve = get_threshold_curve(sample_len,threshold_map)

        plt.plot(threshold_curve)
        plt.plot(predict_labels)
        plt.legend(['threshold curve','predict_labels'])
        plt.show()
        print(sample_no,len(threshold_curve),len(predict_labels))
        statistic_value, confusion_matrix = evaluate_sample(true_labels_all[sample_no],predict_labels,threshold_curve)
        precision_all.append(statistic_value[0])
        recall_all.append(statistic_value[1])
        f1_all.append(statistic_value[2])
        TP_all += confusion_matrix[0]
        TN_all += confusion_matrix[1]
        FP_all += confusion_matrix[2]
        FN_all += confusion_matrix[3]
    plt.plot(precision_all)
    plt.plot(recall_all)
    plt.plot(f1_all)
    plt.legend(['Precision','Recall','F1'])
    plt.show()
def main():
    evaluate_using_threshold_curve(count = 3)
if __name__ == '__main__':
    main()