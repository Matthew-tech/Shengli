#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-15 下午8:39
# @Author  : zejin
# @File    : error_bar.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_error_bar_model_selection():
    error = 'CEE'
    layers_first = True
    sourcefile = '../data_prepare/point_to_label/result_1_6/results_'+error+'.csv'
    data = pd.read_csv(sourcefile)
    trace_range = data['trace_range']

    bar_data = {}
    for i in range(len(trace_range)):
        if data.iloc[i][3] not in bar_data:
            bar_data[data.iloc[i][3]] = [data.iloc[i][7]]
        else:
            bar_data[data.iloc[i][3]].append(data.iloc[i][7])
    size = len(bar_data)
    x = np.arange(size)
    total_width, n = 0.2, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    #x = [0.0,1.0,2.0]
    #exit()
    if layers_first:
        labels = ['layer_1 GN','layer_1 MN','layer_1 None',
                  'layer_2 GN','layer_2 MN','layer_2 None',
                  'layer_3 GN','layer_3 MN','layer_3 None',]
    else:
        labels = ['layer_1 GN','layer_2 GN','layer_3 GN',
                  'layer_1 MN','layer_2 MN','layer_3 MN',
                  'layer_1 None','layer_3 None','layer_3 None',]

    for i in range(len(bar_data.get(0))):
        data = []
        for key in sorted(bar_data.keys()):
            data.append(bar_data[key][i])
        plt.bar(x+i*width,data,width=width,label = labels[i])
    #plt.bar(x, a, width=width, label='a')
    #plt.bar(x + width, b, width=width, label='b')
    #plt.bar(x + 2 * width, c, width=width, label='c')
    plt.xticks([0,1,2],['trace_range 0','trace_range 1','trace_range 2'])
    if error == 'CEE':
        error = 'Cross Entropy Loss'
    else:
        error = 'Mean Square Error'
    plt.ylabel(error)
    plt.title(error+' in validation data set')
    plt.legend(loc='upper right')
    plt.show()
def plot_error_bar_model_selection_further():

    trace_range = 0

    source_file_c = '../model_training/results_clarified_trace_range_'+str(trace_range)+'.csv'
    source_file = '../model_training/results_trace_range_'+str(trace_range)+'.csv'
    data = pd.read_csv(source_file)
    data_clarified = pd.read_csv(source_file_c)
    """
    layers = data['layers'].values
    layers_map = {}
    for index, layer in enumerate(layers):
        if layer in layers_map:
            layers_map[layer].append(index)
        else:
            layers_map[layer] = [index]
    size = len(layers_map[1])
    x = np.arange(size)
    total_width, n = 0.2, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    for i,key in enumerate(sorted(layers_map.keys())):
        bar_data = []
        for index in layers_map[key]:
            bar_data.append(data['error'].iloc[index])
        plt.bar(x+i*width,bar_data,width=width)
    plt.show()
    """
    bar_data = data['error'].values
    bar_data_c = data_clarified['error'].values
    plt.bar(range(len(bar_data)),bar_data,width=0.4)
    plt.bar(np.arange(len(bar_data_c))+0.4,bar_data_c,width=0.4)
    labels = ['layer_1_GN_cellsize_16','layer_1_GN_cellsize_32',
                'layer_2_GN_cellsize_16','layer_2_GN_cellsize_32',
                'layer_1_MN_cellsize_16','layer_1_MN_cellsize_32',
                'layer_2_MN_cellsize_16','layer_2_MN_cellsize_32']
    plt.legend(['No clarified','Clarified'],fontsize = 10)
    plt.xticks(np.arange(len(bar_data))+0.3,labels,rotation=-20)
    plt.ylabel('Cross Entropy Loss')
    save_file = 'Cross Entropy Loss(trace_range: %g)'%trace_range
    plt.title('Cross Entropy Loss of different params(trace_range: %g)'%trace_range)
    plt.show()
    #plt.savefig()
if __name__ == '__main__':
    # plot_error_bar_model_selection()
    plot_error_bar_model_selection_further()