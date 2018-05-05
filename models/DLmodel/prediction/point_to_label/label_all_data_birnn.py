#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-24 下午8:09
# @Author  : Eric
# @File    : label_all_data_birnn.py

# 71 190

import tensorflow as tf
import os
import datetime
import numpy as np
sampling_points = 1251
import struct
from Configure.global_config import *
from models.DLmodel.model.point_to_label.Config import file_loc_gl
from data_prepare.select_high_correlation_attrs import check_is_high_correlation    # 判断是否是高相关性的文件
import pickle
import csv
import sys
from Results.result_modify import add_trace_head

def print2csv(mode,rate,content=None):
    with open(os.path.join(file_loc_gl.infopresent,mode+'_info.csv'),'w') as file:
        writer = csv.writer(file,lineterminator='\n')
        writer.writerow([content,rate])
# b_t_dict 的key为line_no 和 cdp_no, value为时间表示的顶底
with open('data/2-bottom_top_files/b_t_dict.pkl','rb') as file:
    bt_dict = pickle.load(file)

# weights_tracerange_0_layer_2_norm_MN_cell_16_dropout_0.3_GRU_ts_False.index
trace_range = None
layer = None
norm = None
cellsize = None
dropout = None
cell_type = None       # 1表示使用目标层段的label，4表示使用所有的标记label
use_ts = None

paras_dir = 'Results/point_to_label/BiRNN/best_paras'
with open(os.path.join(paras_dir, 'best_paras.pkl'), 'rb') as f:
    paras = pickle.load(f)

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
"""
trace_range = 0
layer = 2
norm = 'MN'
cellsize = 16
dropout = 0.3
cell_type = 'GRU'
use_ts = 'True'
"""
if None in [trace_range,layer,norm,cellsize,dropout,cell_type,use_ts]:
    print('参数还未设置')
    exit()
predict_param_com = 'tracerange_%g_layer_%g_norm_%s_cell_%g_dropout_%g_%s_ts_%s'\
                    %(trace_range,layer,norm,cellsize,dropout,cell_type,use_ts)
# 读取归一化用的文件
norm_dir ='data/full_train_data/'
norm_filename = 'max_min_mean_std_new.pkl'

with open(os.path.join(norm_dir,norm_filename), 'rb') as file:
    norm_paras = pickle.load(file)      # [max, min, mean, std]

line_skip = 2
attr_num = 76

def get_files():
    cube_dir = file_loc_gl.seismic_sgy_file_path_base#修改
    #cube_dir = 'data/seismic_data/'
    file_list = []
    for child_dir in os.listdir(cube_dir):
        for file in os.listdir(os.path.join(cube_dir,child_dir)):
            if not check_is_high_correlation(child_dir,child_dir+'-'+file)[0]: continue
            file_list.append([file,os.path.join(cube_dir,child_dir),child_dir])
    #return file_list * attr_num
    return sorted(file_list, key=lambda x:x[0])

def get_input(file_list, trace_co,read_num = 1,rate = 0,head_only=False):
    """
    file_list 为已经排好序的 76 个地震体文件
    :param file_list:
    :param trace_co: 需要抽取的第 trace_no 个道数据
    :param raed_num 一次性读取的道数
    :param head_only 是否只读取头部信息
    :return:
    """
    X_all = []
    file_no = 0
    attr_specified = 'CDD_bigdata'  # 将原始振幅数据的 卷头和道头加入到预测的地震中
    #print(len(file_list),file_list)
    for filepath in file_list:      # filepath[0] 为文件名
        with open(os.path.join(filepath[1],filepath[0]),'rb') as file:
            if attr_specified in filepath[0]:
                volumn_head = file.read(3600)  # 跳过道头
                trace_head_all = []
            else:
                file.read(3600)
            # 最前面的627 和 628 没有顶底，因此直接跳过
            #file.read((trace_co)* (240 + sampling_points * 4))  # 跳过前面的trace_no，然后读取后面的道
            bytes_skip = (trace_co) * (240 + sampling_points * 4)
            # file.seek(0)
            file.seek(bytes_skip,1)
            X = []
            for read_no in range(read_num):
                trace_head = file.read(240)
                if attr_specified in filepath[0]:
                    trace_head_all.append(trace_head)
                Cur_trace_data = []
                for point_i in range(sampling_points):
                    Cur_trace_data.append(struct.unpack('!f', file.read(4))[0])
                # 对Cur_trace_data 进行归一化，使用norm_paras 中的第file_no个参数进行归一化
                Cur_norm = norm_paras[filepath[0]]
                if norm == 'GN':
                    Cur_trace_data = [(point - Cur_norm[2]) / (Cur_norm[3]) for point in Cur_trace_data]
                elif norm == 'MN':
                    Cur_trace_data = [(point-Cur_norm[1])/(Cur_norm[0]-Cur_norm[1]) for point in Cur_trace_data]
                X.append(Cur_trace_data)
            if file_no % 10 == 0:
                print('rate:%g, file_no:%g'%(rate,file_no),predict_param_com)
            file_no += 1
            X_all.append(X)
        if head_only:
            break
    # X_all 为 [76,read_num,1251]
    # 将X_all 变成 [read_num, 1251, 76的形式]
    if head_only: return volumn_head,trace_head_all
    else:    return volumn_head,trace_head_all, np.asarray(X_all).transpose([1,2,0])
def get_input_len(trace_start, read_num = 1):
    """
    获取每个　ｔｒａｃｅ_no 对应的目标层段开始和结束位置
    :param trace_no:    从trace_no 开始读取
    :param read_num: 表示一次读取的道数
    :return:
    """
    reservoir_start = []
    reservoir_end = []
    # 首先根据　 trace_no 得到line_no 和 cdp_no
    for trace_no in range(trace_start,trace_start+read_num):
        line_no = trace_no//(cdp_e-cdp_s+1) + line_s
        cdp_no = trace_no%(cdp_e-cdp_s+1)   + cdp_s
        C_range = bt_dict[str(line_no)+'.'+str(cdp_no)]
        reservoir_start.append(C_range[0])
        reservoir_end.append(C_range[1])
    return [reservoir_start,reservoir_end]
def print2result(trace_range,trace_start,head,trace_scope,predict_labels):
    """
    :param trace_range:     表示使用那个模型进行预测，trace_range 训练出来的模型
    :param trace_no:        表示当前预测的第几个道
    :param head:            head[0] 为卷头，head[1] 为道头
    :param trace_scope:     目标层段的范围，时间表示
    :param predict_labels:  预测的label
    :return:
    """
    res_dir = 'Results/point_to_label/BiRNN/'
    res_file = os.path.join(res_dir,'result_%s'%(predict_param_com))
    with open(res_file,'ab') as file:
        trace_count = 0
        for i,trace_no in enumerate(range(trace_start,trace_start+len(predict_labels))):
            r_len = int(trace_scope[1][trace_count]/2) - int(trace_scope[0][trace_count]/2)
            Cur_labels = predict_labels[trace_count,:r_len,1].reshape([1,-1]).tolist()[0]
            if trace_no == line_skip * (cdp_e - cdp_s + 1):
                file.write(head[0])     # 写入卷头
                file.write(head[1][i])     # 写入道头
            else:
                file.write(head[1][i])
            # 写入预测的数据
            trace = [0] * int(trace_scope[0][trace_count]/2)
            trace.extend(Cur_labels)
            trace.extend([0] * (sampling_points - len(trace)))

            for label in trace:
                file.write(struct.pack('!f',float(label)))
            trace_count += 1
def get_feed_x(X,range_,max_len):
    """
    :param X: 输入，维度为 [read_num, 1251,76]
    :param range_:   每一个read_num 的有效范围，     时间表示
    :param max_len： 采样点表示
    :return: 模型输入，维度为 [read_num, max_len,76],并将其进行归一化
    """
    X_ret = []
    for i in range(len(X)):
        X_ret.append(X[i,int(range_[0][i]/2):int(range_[0][i]/2)+max_len,:])
    return np.asarray(X_ret).reshape([-1,max_len,attr_num])
def predict_all_area(trace_range = trace_range):
    with tf.Session() as sess:
        weight_dir = 'models/models_weight/BiRNN/'#'models/models_weight/BiRNN/'
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
        cdp_num = cdp_e-cdp_s+1             # 表示每个线的道数
        line_num = line_e - line_s + 1      # 一共这么多线
        read_num = 664                      # 表示一次读取100个道
        #for trace_no in range(line_skip*cdp_num,(line_num - 27) * cdp_num):  # 遍历文件中的每一道，trace_no 从 0 开始计数
        calculate_num = 0
        time_s_all = datetime.datetime.now()
        res_dir = 'Results/point_to_label/BiRNN/'
        res_file = os.path.join(res_dir,'result_%s'%(predict_param_com))
        if os.path.exists(res_file+'_mod.sgy'):
            print('预测文件：%s已生成!'%res_file+'_mod.sgy')
            return
        for trace_no in np.arange(line_skip * cdp_num, (line_num - 27) * cdp_num,read_num):  # 遍历文件中的每一道，trace_no 从 0 开始计数
            time_s = datetime.datetime.now()

            rate = calculate_num / (len(np.arange(line_skip * cdp_num, (line_num - 27) * cdp_num, read_num)))
            # 从trace_no 开始读取read_num 道并返回, trace_head 是一个 list
            volumn_head, trace_head,X = get_input(files_list,trace_no,read_num,rate)
            [X_start,X_end] = get_input_len(trace_no,read_num)  # 时间表示
            X_length = [int(x_end/2)-int(x_start/2) for x_end,x_start in zip(X_end,X_start)]#采样点表示
            X = get_feed_x(X,[X_start,X_end],max(X_length))
            feed_dict = {input_x: X, X_len:max(X_length),seq_length: X_length, keep_prob:1}
            # 开始预测
            Cur_predict_labels = sess.run(y, feed_dict=feed_dict)       # [read_num,max_len,2]
            print2result(trace_range,trace_no,[volumn_head,trace_head],[X_start,X_end],Cur_predict_labels)
            calculate_num += 1

            time_e = datetime.datetime.now()
            print('line_exed%g / all_line%g, rate:%g,Cur_time:%s ,all_time:%s'
                  %(calculate_num,line_num-2-27,rate, (time_e-time_s),(time_e-time_s_all)))
            print2csv(mode='predict_all',rate=rate,content='line_exed%g / all_line%g, rate:%g,Cur_time:%s ,all_time:%s'
                  %(calculate_num,line_num-2-27,rate, (time_e-time_s),(time_e-time_s_all)))
        # 对生成的文件进行修改
        add_trace_head(filepath=res_file)
def main():
    predict_all_area(trace_range = 0)
    #get_files()
if __name__ == '__main__':
    main()
