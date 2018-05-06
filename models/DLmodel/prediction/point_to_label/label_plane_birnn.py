#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-20 上午10:50
# @Author  : Eric
# @File    : label_plane_birnn.py
import tensorflow as tf
import os
import pickle
import datetime
import numpy as np
import struct
from Configure.global_config import *
from models.DLmodel.prediction.point_to_label.label_all_data_birnn import get_files, get_input
import matplotlib.pyplot as plt
import csv
"""
cdp_s = 1190        # 地震体的cdp_s 是从1189 开始的
cdp_e = 1851        # 地震体的cdp_e 从1852 截止， 因此cdp 数量为 662
line_s = 629
line_e = 2242
sampling_points = 1251
"""
line_skip = 2

trace_range = None
layer = None
norm = None
cellsize = None
dropout = None
cell_type = None       # 1表示使用目标层段的label，4表示使用所有的标记label
use_ts = None

paras_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/best_paras')
with open(os.path.join(paras_dir, 'best_paras.pkl'), 'rb') as f:
    paras = pickle.load(f)
def print2csv(mode,rate,content=None):
    with open(os.path.join(file_loc_gl.infopresent,mode+'_info.csv'),'w') as file:
        writer = csv.writer(file,lineterminator='\n')
        writer.writerow([content,rate])
trace_range = paras['trace_range']
layer = paras['layers']
norm = paras['normalize']
cellsize = paras['cellsize']
dropout = paras['dropout']
cell_type = paras['rnn_cell']
if paras['use_alllabel']:
    use_ts = False
else:
    use_ts = True
if None in [trace_range,layer,norm,cellsize,dropout,cell_type,use_ts]:
    print('参数还未设置')
    exit()
predict_param_com = 'tracerange_%g_layer_%g_norm_%s_cell_%g_dropout_%g_%s_ts_%s'\
                    %(trace_range,layer,norm,cellsize,dropout,cell_type,use_ts)
# 读取归一化用的文件
norm_dir =os.path.join(file_loc_gl.data_root,'full_train_data/')
norm_filename = 'max_min_mean_std_new.pkl'

with open(os.path.join(norm_dir,norm_filename), 'rb') as file:
    norm_paras = pickle.load(file)      # [max, min, mean, std]

def predict_point(line_num = 629, cdp_num = 1190, time = 1100):
    """
    对某一个特定的点进行预测
    :param line_num:  线号
    :param cdp_num:   道号
    :param time:      深度时间
    :return:          预测的label
    """
    with tf.Session() as sess:
        weight_dir = file_loc_gl.weights_BiRNN
        saver = tf.train.import_meta_graph(os.path.join(weight_dir, 'weights_%s.meta'%predict_param_com))
        saver.restore(sess, weight_dir + 'weights_%s'%predict_param_com)
        y = tf.get_collection('predict_network')[0]
        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('input_x').outputs[0]
        X_len = graph.get_operation_by_name('X_len').outputs[0]
        seq_length = graph.get_operation_by_name('seq_length').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        time_s = datetime.datetime.now()
        files_list = get_files()
        trace_no = (int(line_num) - int(line_s)) * (int(cdp_e) - int(cdp_s) + 1) + int(cdp_num) - int(cdp_s)
        volumn_head, trace_head, X = get_input(files_list, trace_no, 1)  # X:shape = (1, 1251, 76)
        # 时间表示, 每一个深度都为2ms, 对于预测平面，返回 trace_no +1 : trace_no +1 + read_num -1
        [X_start, X_end] = [[time], [time+2]]

        X_length = [1]  # 采样点表示
        X = np.asarray(X[0,int(time/2),:]).reshape(1,1,len(X[0,0,:]))
        feed_dict = {input_x: X, X_len: max(X_length), seq_length: X_length, keep_prob: 1}
        # 开始预测
        Cur_predict_labels = sess.run(y, feed_dict=feed_dict)  # [1,1,2]
        predict_label = Cur_predict_labels[0,0,1]
        with open(os.path.join(file_loc_gl.results_root,'predict_point.txt'), 'w') as file:
            file.write(str(predict_label))
        print(predict_label)
        return predict_label

def save_into_file(plane_file, predict_labels):
    """
    将predict_labels 保存到一个新文件中
    :param plane_file:
    :param predict_labels:
    :return:
    """
    with open(plane_file,'r') as file1, open(plane_file+'_predict','w') as file2:
        line = 'tmp'
        while line:
            line = file1.readline()
            line_split = [value for value in line[:-1].split(' ') if value != '']
            key = '%s-%s'%(str(line_split[4],str(int(float(line_split[3])))))
            if key in predict_labels.keys():
                line_add_label = line + ' '*12 + str(predict_labels[key]) + '\n'
            else:
                line_add_label = line
            file2.write(line_add_label)
def plot_predicted_plane(pkl_file,name):
    """
    将预测的值展示的图片上，并将图片保存下来
    :param predict_labels:
    :return:
    """
    fig_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/pics')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    figpath = os.path.join(fig_dir,'%s_predict_plane.png'%name)
    with open(pkl_file,'rb') as file:
        data = pickle.load(file)
        predict_labels = []
        for key in data.keys():
            line_cdp = key.split('-')
            Cur_label = [int(line_cdp[0]),int(line_cdp[1])]
            predict_labels.append(Cur_label+[data.get(key)])
        predict_labels = sorted(predict_labels,key = lambda x:(x[0],x[1]))
        print(len(predict_labels))
        labels = [label[2] for label in predict_labels]
        labels = np.asarray(labels).reshape((-1,662))
        label_max = np.max(labels)
        label_min = np.min(labels)
        for i in range(len(labels)):
            for j in range(len(labels[0,:])):
                labels[i,j] = (labels[i,j]-label_min)/(label_max-label_min)
        plt.imshow(labels)
        plt.colorbar()
        
        plt.savefig(figpath,dpi=1000)

# 使用 BiRNN 模型对一个平面进行预测
def predict_plane(plane_file='{}/plane_loc/ng32sz_grid_28jun_154436.p701'.format(file_loc_gl.data_root),
                  name='ng32sz_grid_28jun_154436.p701'):
    """
    对某一特定层位进行预测, 并将结果存放在Results中
    :param plane_file: 层位所在的文件位置
    :return:
    """
    # 将plane_file 保存道本地的pickle文件中
    plane_file_pkl = plane_file+'.pkl'
    if not os.path.exists(plane_file_pkl):
        with open(plane_file,'r') as file1, open(plane_file_pkl,'wb') as file2:
            line = file1.readline()
            loc_time_dict = {}
            while line and line != []:

                line_split = line[:-1].split(' ')
                line_split = [value for loc, value in enumerate(line_split) if value != '']
                line_num=line_split[4]
                cdp_num=str(int(float(line_split[3])))
                time=line_split[2]
                loc_time_dict[line_num+'-'+cdp_num] = float(time)
                line = file1.readline()
            print(loc_time_dict)
            pickle.dump(loc_time_dict,file2)
    with open(plane_file_pkl,'rb') as file:
        loc_time_dict = pickle.load(file)

    def get_input_len(trace_start, read_num):
        """
        返回每一道的起始和终止位置, 对于预测一个平面来讲，需要返回trace_start + 1 - trace
        :param trace_no: 根据第几道得到 道号和线号
        :param read_num: 读取的道的数量
        :return:
        """
        reservoir_start = []    # 返回的start 和 end 的个数均为 read_num -2
        reservoir_end = []
        # 首先根据　 trace_no 得到line_no 和 cdp_no
        for trace_no in range(trace_start+1, trace_start + read_num-1):
            line_no = trace_no // (cdp_e - cdp_s + 1) + line_s
            cdp_no = trace_no % (cdp_e - cdp_s + 1) + cdp_s
            C_range = loc_time_dict[str(line_no) + '-' + str(cdp_no)]
            reservoir_start.append(C_range)
            reservoir_end.append(C_range+2)
        return reservoir_start, reservoir_end
    def get_feed_x(X,range_,max_len):
        """
        获取适合的模型输入
        :param X:       shape = (read_num, 1251, 76)
        :param range_:  [X_start, X_end]  len(X_start) = read_num -2 ，均为时间表示
        :param max_len: 1
        :return:        X_re , shape = (read_num-2, 1, 76)
        """
        X_re = []
        for i,trace_no in enumerate(range(1,len(X)-1)):
            X_re.append(X[trace_no,int(range_[0][i]/2),:])
        return np.asarray(X_re).reshape([len(range_[0]),1,len(X[0,0,:])])
    with tf.Session() as sess:
        weight_dir = file_loc_gl.weights_BiRNN
        saver = tf.train.import_meta_graph(os.path.join(weight_dir, 'weights_%s.meta'%predict_param_com))
        saver.restore(sess, weight_dir + 'weights_%s'%predict_param_com)
        y = tf.get_collection('predict_network')[0]
        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('input_x').outputs[0]
        X_len = graph.get_operation_by_name('X_len').outputs[0]
        seq_length = graph.get_operation_by_name('seq_length').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        time_s = datetime.datetime.now()
        files_list = get_files()
        cdp_num = cdp_e-cdp_s+1             # 表示每个线的道数， 对于一个平面来说变成了662道
        line_num = line_e - line_s + 1      # 一共这么多线
        read_num = cdp_num                      # 表示一次读取两条线对应的道数
        calculate_num = 0
        time_s_all = datetime.datetime.now()
        predict_labels = {}
        if os.path.exists('%s_predict_labels.pkl'%name):
            print('文件%s已存在，如需重新预测，请先删除！'%('%s_predict_labels.pkl'%name))
            return 
        for trace_no in np.arange(line_skip * cdp_num, (line_num - 27) * cdp_num,read_num):  # 遍历文件中的每一道，trace_no 从 0 开始计数
            time_s = datetime.datetime.now()

            rate = calculate_num / (len(np.arange(line_skip * cdp_num, (line_num - 27) * cdp_num, read_num)))
            # 从trace_no 开始读取read_num 道并返回, trace_head 是一个 list
            volumn_head, trace_head,X = get_input(files_list,trace_no,read_num,rate)     # X:shape = (read_num, 1251, 76)
            # 时间表示, 每一个深度都为2ms, 对于预测平面，返回 trace_no +1 : trace_no +1 + read_num -1
            [X_start,X_end] = get_input_len(trace_no,read_num)
            X_length = [int(x_end/2)-int(x_start/2) for x_end,x_start in zip(X_end,X_start)]#采样点表示
            X = get_feed_x(X,[X_start,X_end],max(X_length))
            feed_dict = {input_x: X, X_len:max(X_length),seq_length: X_length, keep_prob:1}
            # 开始预测
            Cur_predict_labels = sess.run(y, feed_dict=feed_dict)       # [read_num,max_len,2]
            for i,trace_start in enumerate(range(trace_no+1, trace_no + read_num-1)):
                line_no = trace_start // (cdp_e - cdp_s + 1) + line_s
                cdp_no = trace_start % (cdp_e - cdp_s + 1) + cdp_s
                predict_labels['%s-%s'%(str(line_no),str(cdp_no))] = Cur_predict_labels[i,0,1]
            calculate_num += 1

            time_e = datetime.datetime.now()
            print('line_exed%g / all_line%g, rate:%g,Cur_time:%s ,all_time:%s'
                  %(calculate_num,line_num-2-27,rate, (time_e-time_s),(time_e-time_s_all)))
            print2csv(mode='predict_plane',rate=rate,content='line_exed%g / all_line%g, rate:%g,Cur_time:%s ,all_time:%s'
                  %(calculate_num,line_num-2-27,rate, (time_e-time_s),(time_e-time_s_all)))
        label_pre_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/plane_prediction')
        if not os.path.exists(label_pre_dir):
            os.makedirs(label_pre_dir)
        
        with open(os.path.join(label_pre_dir,'%s_predict_labels.pkl'%name), 'wb') as pklfile:
            pickle.dump(predict_labels,pklfile,-1)
        # 将预测的平面图图示出来,并保存到本地
        plot_predicted_plane(os.path.join(label_pre_dir,'%s_predict_labels.pkl'%name),name)
        # 将预测的结果写入文件
        save_into_file(plane_file, predict_labels)
        
