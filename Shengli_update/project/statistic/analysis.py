#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-17 下午7:28
# @Author  : zejin
# @File    : analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import pickle
def statistic_num_of_reservoir():
    count = 1
    souceFile = '../data_prepare/point_to_label/results_' + str(count) + '.csv'
    data = pd.read_csv(souceFile)
    samples_indexes = data['No']
    samples_start_index = [loc for loc, i in enumerate(samples_indexes) if i == 1]
    partation = []
    for index, value in enumerate(samples_start_index): # 遍历每个样本
        if index == len(samples_start_index)-1:
            Cur_data = data.iloc[value:]
        else:
            Cur_data = data.iloc[value:samples_start_index[index + 1]]
        true_labels = Cur_data['true_label']
        partation.append(sum(true_labels)/len(true_labels))
    plt.plot(partation)
    plt.xlabel('sample No')
    plt.ylabel('Partation')
    plt.title('Partation of reservoir in each sample')
    plt.show()
    print(sum(data['true_label']/len(data['true_label'])))
def get_loc(true_labels):
    res = [0]
    cur_label = true_labels[0]
    for i,label in enumerate(true_labels[1:]):
        if cur_label!= label:
            res.append(i+1)
            cur_label = label
    return res + [len(true_labels)-1]

def get_labels(count, return_depth = False , des_dir = '../model_training/results_further/'):
#    souceFile = '../data_prepare/point_to_label/result_1_6/results_' + str(count) + '.csv'
    souceFile = os.path.join(des_dir,'results_'+str(count)+'_add_depth.csv')
    data = pd.read_csv(souceFile)
    samples_indexes = data['No']
    samples_start_index = [loc for loc, i in enumerate(samples_indexes) if i == 1]      # 每个样本的初始位置
    true_labels_all = []
    predict_labels_all = []
    depth_all = []
    for index, value in enumerate(samples_start_index):
        if index == len(samples_start_index) - 1:
            Cur_data = data.iloc[value:]
        else:
            Cur_data = data.iloc[value:samples_start_index[index + 1]]
        true_labels = Cur_data['true_label'].values
        predict_labels = Cur_data['predict_label'].values
        depth = Cur_data['depth'].values
        true_labels_all.append(true_labels)
        predict_labels_all.append(predict_labels)
        depth_all.append(depth)
    if return_depth: return true_labels_all,predict_labels_all,depth_all
    else:    return true_labels_all, predict_labels_all
def plot_true_predict_label(count = 2):
    true_labels_all, predict_labels_all = get_labels(count=count,des_dir='../Results/test_wells_result')
    # 找到0和1变化的位置
    sample_count = 1
    for true_labels, predict_labels in zip(true_labels_all,predict_labels_all):

        interval_loc = get_loc(true_labels)
        plt.plot(predict_labels)
        # plt.plot(true_labels)
        # 画出预测值
        plt.plot(interval_loc, predict_labels[interval_loc])
        # 从图中画出间隔虚线
        for i, j in enumerate(interval_loc):
            # 画出间隔，虚线表示
            plt.plot([j, j], [predict_labels[j] - 0.1, predict_labels[j] + 0.1], 'k--', LineWidth='1')
            # 将真实值画出来，0和1
            if i < len(interval_loc) - 1:
                if true_labels[j] == 0:
                    value_y = predict_labels[j] - 0.1
                    color = 'r'
                else:
                    value_y = predict_labels[j] + 0.1
                    color = 'g'
                plt.plot([j, interval_loc[i + 1]], [value_y, value_y], color)
        plt.xlabel('Time(ms)')
        plt.ylabel('True label and predict label')
        plt.title('Sample:%g - Comparison between true label and predict label(yellow line is thresholds)'%sample_count)
        plt.show()
        sample_count += 1
def get_fit_curve(temp_map):
    x_list = []
    y_list = []
    for key in temp_map.keys():
        x_list.append(key)
        y_list.append(sum(temp_map[key])/len(temp_map[key]))
    x_list_sorted = sorted(x_list)
    x_list_index = [x_list.index(i) for i in x_list_sorted]
    y_list = [y_list[i] for i in x_list_index]
    return [x_list_sorted,y_list]
def fit_threshold_curve(count=1,use_depth = False, params = None):
    # true_labels_all 和 predict_labels_all 都是list的list
    if use_depth:
        true_labels_all, predict_labels_all,depth_all = get_labels(count=count,return_depth=use_depth)
    else:
        true_labels_all, predict_labels_all = get_labels(count=count, return_depth=use_depth)
    # 找到0和1变化的位置
    sample_count = 0
    # 记录每个样本的长度和其对应的阈值线的坐标
    threshold_map = {}  # key: 曲线长度 value：对应该长度的点坐标
    for true_labels, predict_labels in zip(true_labels_all,predict_labels_all):
        samle_len = len(true_labels)
        interval_loc = get_loc(true_labels)
        if use_depth:   x_range = np.asarray(interval_loc) + +depth_all[sample_count][0]
        else: x_range = interval_loc
        if samle_len in threshold_map:
            threshold_map[samle_len].append([zip(interval_loc,predict_labels[interval_loc].tolist()),samle_len])
        else:
            threshold_map[samle_len] = [[zip(interval_loc,predict_labels[interval_loc].tolist()),samle_len]]
        sample_count += 1
    fit_func_map = {}
    for key in sorted(threshold_map.keys())[1:]:
        x_value = []
        y_value = []
        temp_map = {}                           # 保存x坐标和y坐标
        for sample_zip in threshold_map[key]:
            x_temp = []
            y_temp = []
            for x, y in sample_zip[0]:
                x_temp.append(x)
                y_temp.append(y)
                if params['type'] != 'Mean':
                    x_value.append(x)
                    y_value.append(y)
                #plt.plot(x_temp, y_temp)
                if x not in temp_map:
                    temp_map[x] = [y]
                else:
                    temp_map[x].append(y)
        if params['type'] == 'polynomial':

            f = np.polyfit(x_value, y_value, params['cishu'])
            p = np.poly1d(f)
            print(p)
            plt.scatter(x_value, y_value)
            plt.plot(list(range(threshold_map[key][0][1])), p(list(range(threshold_map[key][0][1]))))
            plt.show()
            fit_func_map[key] = p
        elif params['type'] == 'Mean':
            fit_func_map[key] = get_fit_curve(temp_map)         # 是一个[[x 的坐标],[y的坐标]]
            #plt.plot(fit_func_map[key][0],fit_func_map[key][1],'k--')
            #plt.show()
    if not os.path.exists('./threshold_curve'):
        os.makedirs('./threshold_curve')
    with open('./threshold_curve/threshold_curve_'+str(len(fit_func_map))+'_count_'+str(count)+'.pkl','wb') as file:
        pickle.dump(fit_func_map,file,-1)

def calculate_acc_using_segment_thres(count = 1):
    """
    :param count: 表示第几个result 文件
    :return:
    """
    true_labels_all, predict_labels_all = get_labels(count = count)
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    TP_all, FN_all, TN_all,FP_all = 0,0,0,0
    for true_labels, predict_labels in zip(true_labels_all, predict_labels_all):
        interval_loc = get_loc(true_labels)
        TP, TN, FP, FN = 0,0,0,0
        for i,j in enumerate(interval_loc[:-1]):
            k = (predict_labels[j] - predict_labels[interval_loc[i+1]])/(j-interval_loc[i+1])
            b = predict_labels[j] - k*j
            for point in range(j,interval_loc[i+1]):    # point表示横坐标
                Cur_thres = k*point + b
                if true_labels[point] == 1 and predict_labels[point] >= Cur_thres:
                    TP += 1
                    TP_all += 1
                elif true_labels[point] == 1 and predict_labels[point] < Cur_thres:
                    FN += 1
                    FN_all += 1
                elif true_labels[point] == 0 and predict_labels[point] < Cur_thres:
                    TN += 1
                    TN_all += 1
                else:
                    FP += 1
                    FP_all += 1
        acc = (TP+TN)/ (TP+TN+FP+FN); acc_list.append(acc)
        recall = TP/(TP+FN)         ;recall_list.append(recall)
        precision = TP/(TP+FP)      ;precision_list.append(precision)
        F1 = 2*recall*precision/(recall+precision) ;f1_list.append(F1)
    acc = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all);
    recall = TP_all/(TP_all + FN_all)
    precision = TP_all/(TP_all+FP_all)
    f1 = 2*recall*precision/(recall+precision)
    return [acc_list, recall_list, precision_list, f1_list],[acc,recall,precision,f1]
def plot_each_result(result_list):
    for stat in result_list:
        plt.plot(stat)
    plt.legend(['Acc','Recall','Precision','F1'])
    plt.xlabel('Sample No')
    plt.ylabel('Rate')
    plt.title('ACC, Reall, Precision and F1 score')
    plt.show()
if __name__ == '__main__':
#    statistic_num_of_reservoir()           # 统计每个井储层个数所占的比例
    plot_true_predict_label(count=4)    # 可视化预测label和真实label
#    result_sample,result_all = calculate_acc_using_segment_thres(count = 1)     # 使用分段线性差值的结果作为threshold，并计算每个样本的准确率
#    plot_each_result(result_sample)
#    print(result_all)
    """
    fit_param = {'type':'polynomial','cishu':5}     # 多项式拟合
    fit_param = {'type':'Mean'}
    fit_threshold_curve(count=3,use_depth=False,params = fit_param)
    """