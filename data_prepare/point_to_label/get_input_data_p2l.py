#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-13 下午12:50
# @Author  : Eric
# @File    : get_input_data_p2l.py
import numpy as np
import os
import pickle
import pandas as pd
import random
from Configure.global_config import file_loc_gl
from data_prepare.select_high_correlation_attrs import check_is_high_correlation    # 判断是否是高相关性的文件
from data_prepare.point_to_label.data_util import Build_Interpolation_function        # 构建插值函数
from data_prepare.point_to_label.data_util import get_well_name                       # 建立井的位置和名称的map

import matplotlib.pyplot as plt
class BatchGenerator_p2l(object):
    def __init__(self, input_x, input_y , batch_size,ts_max):
        """
        :param input_x: list, 其中每个元素为一个array,表示每一道的feature
        :param input_y: list, 其中每个元素为一个list，与input_x 相对应
        :param batch_size:
        :param ts_max: 把所有的输入都补全成了 ts_max 的长度
        """
        self._input_x = input_x
        self._input_y = input_y
        self._samples_num = len(self._input_x)
        self._seq_len = [len(self._input_x[i]) for i in range(self._samples_num)]
        self._seq_len_max = max(self._seq_len)
        self._feature_dim = len(self._input_x[0][0])
        # 将self_input_x 和 self._input_y 都补全成 max的长度
        self._input_x = [np.vstack((self._input_x[i],np.zeros(shape=[self._seq_len_max-len(self._input_x[i]),self._feature_dim])))
                         for i in range(self._samples_num)]
        self._input_y = [self._input_y[i]+[0]*(self._seq_len_max-len(self._input_y[i])) for i in range(self._samples_num)]
        random_index = list(range(self._samples_num))
        random.shuffle(random_index)
        self._samples_index = random_index
        # 将样本打乱
        self._input_x = [self._input_x[i] for i in self._samples_index]
        self._input_y = [self._input_y[i] for i in self._samples_index]
        self._seq_len = [self._seq_len[i] for i in self._samples_index]
        self._batch_size = batch_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def samples_num(self):
        return self._samples_num
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def epochs_completed(self):
        return self._epochs_completed
    @property
    def samples_index(self):
        return self._samples_index
    @property
    def seq_len_max(self):
        return self._seq_len_max
    @property
    def feature_dim(self):
        return self._feature_dim
    def one_hot(self,labels):
        for i,sample in enumerate(labels):
            labels[i] = [[0,1] if label == 1 else [1,0] for label in sample]
        return labels
    def next_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += self._batch_size
        if self._index_in_epoch > self._samples_num:
            # finish epoch
            self._epochs_completed +=1

            # shuffle the data
            data_index = list(range(self._samples_num))
            np.random.shuffle(data_index)
            self._samples_index = data_index
            # 将样本打乱
            self._input_x = [self._input_x[i] for i in self._samples_index]
            self._input_y = [self._input_y[i] for i in self._samples_index]
            self._seq_len = [self._seq_len[i] for i in self._samples_index]
            start = 0
            self._index_in_epoch = self._batch_size
            assert self._batch_size <= self._samples_num
        end = self._index_in_epoch
        feature = np.asarray(self._input_x[start:end])
        label = np.asarray(self.one_hot(self._input_y[start:end]))
        seq_len = self._seq_len[start:end]
        return {'feature':feature,'label':label,'seq_len':seq_len}
def get_norm_param(sourcefile = os.path.join(file_loc_gl.data_root,'full_train_data/max_min_mean_std_new.pkl')):
    with open(sourcefile,'rb') as file:
        param_dict = pickle.load(file)      # {filename:[max, min, mean, std] }
    return param_dict
norm_file = os.path.join(file_loc_gl.data_root,'full_train_data/max_min_mean_std_new.pkl')
#with open(norm_file, 'rb') as file:
    #param_dict = pickle.load(file)  # {filename:[max, min, mean, std] }

def norm(seismic_data_map, Cur_file,type = '',trace_range = 0):
    """
    对整个地震体数据进行归一化
    :param seismic_data_map:    表示抽取出来的地震数据，key 为井的名字，value为trace data
    :param Cur_file:            当前处理的文件名 [filename, path ,child_dir]
    :param type:    归一化方法， ‘’ 表示不进行归一化，‘GN’：高斯归一化,'MN' :最大最小归一化
    :return:
    """
    paras = []      # 记录最大值，最小值，均值和方差等
    selected_points = (trace_range * 2 + 1) ** 2
    trace_i = [24 - int(selected_points / 2), 24 + int(selected_points / 2) + 1]
    trace_i = range(trace_i[0],trace_i[1])
    Cur_filename = Cur_file[0][Cur_file[0].index(Cur_file[2])+len(Cur_file[2])+1:-4]
    param_dict = get_norm_param(sourcefile = norm_file)
    Cur_norm_param = param_dict[Cur_filename]   # [max, min, mean, std]
    if type == 'None':
        return seismic_data_map
    else:
        if type == 'GN':
            mean = Cur_norm_param[2]
            std = Cur_norm_param[3]
            for key in seismic_data_map.keys():         # key 为井的坐标
                for i in range(len(seismic_data_map[key])):
                    seismic_data_map[key][i] = [(seismic_data_map[key][i][j]-mean)/std for j in range(len(seismic_data_map[key][i]))]
        if type == 'MN':
            min_value = Cur_norm_param[1]
            max_value = Cur_norm_param[0]
            paras = [min_value, max_value]
            bias = max_value - min_value
            for key in seismic_data_map.keys():
                for i in range(len(seismic_data_map[key])):
                    seismic_data_map[key][i] = [(seismic_data_map[key][i][j]-min_value)/bias for j in range(len(seismic_data_map[key][i]))]
        return seismic_data_map, paras
#获取属性文件的list （可选择去掉高相关性的特征）
def get_files_list(feature_file_dir=file_loc_gl.full_train_data):
    """
    获取经过排序的特征文件夹  
    :param feature_file_dir:
    :return:
    """
    attr_file_list = []
    for child_dir in os.listdir(feature_file_dir):
        if not os.path.isdir(os.path.join(feature_file_dir, child_dir)): continue
        Cur_file_dir = os.path.join(feature_file_dir, child_dir)
        for filename in os.listdir(Cur_file_dir):
            if os.path.isdir(os.path.join(Cur_file_dir, filename)): continue
            # 判断是否是高相关性文件
            if not check_is_high_correlation(child_dir, filename)[0]: continue
            attr_file_list.append([filename, Cur_file_dir,child_dir])
    return attr_file_list
def get_seismic_data(trace_range = 0, normalize = 'GN'):
    """
    获取所有井对应的地震数据，每个地震数据的维度为 76 * sampling_points
    :param trace_range:
    :param normalize:
    :return:
    """
    print('正在读取地震数据...')
    selected_points = (trace_range * 2 + 1) ** 2
    trace_i = [24 - int(selected_points / 2), 24 + int(selected_points / 2) + 1]
    trace_i = range(trace_i[0], trace_i[1])
    all_seismic_data = {}
    feature_file_dir = file_loc_gl.full_train_data
    attr_count = 0

    attr_file_list = get_files_list(feature_file_dir)
    for Cur_file in sorted(attr_file_list, key=lambda x: x[0][x[0].index(x[2])+len(x[2])+1:]):

        with open(os.path.join(Cur_file[1], Cur_file[0]), 'rb') as file:
            trace_range_record = pickle.load(file)
            if normalize !='None':
                Cur_attr_all_data,paras = norm(pickle.load(file),Cur_file, type=normalize, trace_range=trace_range)  # 对每个地震体数据进行归一化
            else:
                Cur_attr_all_data = pickle.load(file)

            for key in Cur_attr_all_data.keys():
                Cur_wellname = get_well_name(int(float((key.split(',')[1]))), int(float(key.split(',')[0])))
                if len(Cur_attr_all_data.get(key)) != 49: continue
                if Cur_wellname is None: continue
                if Cur_wellname not in all_seismic_data:
                    all_seismic_data[Cur_wellname] = [np.array(Cur_attr_all_data.get(key)[i]).T for i in trace_i]
                else:
                    all_seismic_data[Cur_wellname] = \
                        [np.vstack((all_seismic_data[Cur_wellname][i], np.array(Cur_attr_all_data.get(key)[j]).T)) for
                         i, j
                         in enumerate(trace_i)]
        attr_count += 1
        if attr_count % 10 == 0:
            print('file_no:%g / ALL:%g'%(attr_count,len(attr_file_list)))

    # 将map 中的每个val进行转置
    for key in all_seismic_data.keys():
        all_seismic_data[key] = [all_seismic_data[key][i].T for i in range(len(trace_i))]
    return all_seismic_data
def clarify_labels(all_labels,C = 8):
    """
    将获取的目标层段的label进行净化，如果出现了单个0或者单个1，而前后则有连续的1或0（C/2个），则将其置为相反的label
    :param labels_ts:  目标层段的label
    :return:
    """
    interval_ = [-2,-1,1,2]
    for key in all_labels.keys():
        Cur_labels = all_labels[key][1]
        for i in range(len(Cur_labels)-C):
            find_same_label = False
            if sum(Cur_labels[i:i+C]) == C-1:   # 表示当前窗口中只有一个 0
                flag = 0
                flag_loc = Cur_labels[i:i+C].index(0)
                loc_in_label = i + flag_loc
            elif sum(Cur_labels[i:i+C]) == 1:   # 表示当前窗口中只有一个 1
                flag = 1
                flag_loc = Cur_labels[i:i+C].index(1)
                loc_in_label = i + flag_loc
            else:
                continue
            for index in interval_:             # 如果在前后 +-2的范围内发现了相同的label，则跳过
                if len(Cur_labels)>loc_in_label+interval_[index] >=0:
                    if Cur_labels[loc_in_label+interval_[index]] == flag:
                        find_same_label = True ; break
            if find_same_label: continue
            else: Cur_labels[loc_in_label] = 1-flag
        all_labels[key] = [all_labels[key][0],Cur_labels]
    return all_labels

def get_labels(clarify = True,use_alllabels=False):
    """
    :param clarify: 是否删除中间出现的单个 0 或者单个 1
    :return:
    """
    print('正在提取井的标记信息...')
    all_labels = {}
    if use_alllabels:
        sourceFile = file_loc_gl.well_reservoir_Info_all_clean
    else:
        sourceFile = file_loc_gl.well_reservoir_Info_clean
    reservoir_labels = pd.read_csv(sourceFile)
    well_names = reservoir_labels['well_no']
    well_index_map = {}         # key 为 wellname， value 为其对应的行数，是一个list
    for i,Cur_well_name in enumerate(well_names):
        if Cur_well_name in well_index_map:
            well_index_map[Cur_well_name].append(i)
        else:
            well_index_map[Cur_well_name] = [i]
    for Cur_well in well_index_map.keys():
        labels = reservoir_labels.loc[well_index_map[Cur_well]]
        # 得到当前井的时深转化差值函数
        rel_dir = file_loc_gl.depth_time_rel_dir
        if os.path.exists(os.path.join(rel_dir, Cur_well + '_ck.txt')):
            _,_,f = Build_Interpolation_function(os.path.join(rel_dir, Cur_well + '_ck.txt'))
        elif os.path.exists(os.path.join(rel_dir, Cur_well.lower() + '_ck.txt')):
            _,_,f = Build_Interpolation_function(os.path.join(rel_dir, Cur_well.lower() + '_ck.txt'))
        else:
            print('没有找到井：', Cur_well, '时深转化文件....')
            continue
        # 取出顶底对应的位置
        top = f(float(labels.iloc[0][1]))
        bottom = f(float(labels.iloc[-1][2]))
        Cur_label = []
        for i in labels.index:
            #print(i,int(f(labels.iloc[i].values[1])/2),int(f(labels.iloc[i].values[2])/2))
            for j in range(int(f(labels.loc[i].values[1])/2),int(f(labels.loc[i].values[2])/2)):
                Cur_label.append(int(labels.loc[i].values[3]))
        all_labels[Cur_well] = [[top,bottom], Cur_label]       # 分别是目标层段和label值
    if clarify: return clarify_labels(all_labels)
    else:       return all_labels
def get_ts_data(seismic_data,labels,cnnrnn=False):
    """
    获取label 之间的数据
    :param seismic_data:
    :param labels:
    :param cnnrnn: 表示是否用来构造cnn rnn的输入
    :return: ts_max 表示最大的目标层段
    """
    ts_max = 0
    overlapping_keys = set(list(seismic_data.keys()) + list(labels.keys()))
    for key in overlapping_keys:
        if key in seismic_data.keys() and key in labels.keys(): continue
        #print(key)
        seismic_data.pop(key, 1); labels.pop(key,1)
    for key in seismic_data.keys():
        top = int(labels.get(key)[0][0]/2)
        bottom = int(labels.get(key)[0][1]/2)
        if cnnrnn:
            seismic_data[key] = seismic_data[key][top:bottom,:,:,:]
        else:
            seismic_data[key] = [features[top:bottom,:] for features in seismic_data[key]]
        ts_max = max(ts_max, int((bottom-top)))
        #print(seismic_data[key][0].shape,len(labels[key][1]))

    return seismic_data,labels, ts_max
def change(x,y,return_depth = False):
    """
    :param x: 是一个list，list中的每个元素也都是一个list，包含若干feature map，对应features
    :param y: 是一个list，需要根据x 中feaiture map的数量进行扩展，对应labels
    :return:
    """
    X_ret = []
    Y_ret = []
    depth = []
    for i in range(len(x)):
        trace_num = len(x[i])
        for Cur_trace in x[i]:
            X_ret.append(Cur_trace)
            Y_ret.append(y[i][1])
            depth.append([int(y[i][0][0]/2),int(y[i][0][1]/2)])
    if return_depth:    return X_ret,Y_ret, depth       # 返回的深度信息为采样点表示
    else:   return X_ret,Y_ret
def augment_data(labels, seismic_data,each_len = 100,interval = 1):
    """

    :param labels: map_, key: well_name, value: [[top, bottom], labels], top 和 bottom为时间表示
    :param seismic_data: key: well_name, value: ts_len * 76
    :param each_len:    数据扩增后每个样本的长度,           采样点表示
    :param interval:    扩增的时候each_len 划过的长度      采样点表示
    :return:    labels_aug:     key: well_name_No,  value 值形式相同
                seismic_aug:    key: well_name_No   value 值形式相同
    """
    labels_aug = {}
    seismic_data_aug = {}
    for key in sorted(labels.keys()):
        time_start = labels[key][0][1]
        sample_len = int((labels[key][0][1] - labels[key][0][0])/2)
        if sample_len <= each_len:
            labels_aug[key] = labels[key]
            seismic_data_aug[key] = seismic_data[key]
        else:
            Cur_aug_num = int(sample_len-each_len) // interval
            Cur_labels = labels[key][1]
            Cur_seismic_data = seismic_data[key]
            for i in range(Cur_aug_num):
                Cur_aug_key = key+'_%g'%i
                # 对labels 进行扩充
                labels_aug[Cur_aug_key] = [[time_start+i*2,time_start+(i+each_len)*2],Cur_labels[i:i+each_len]]
                # 对seismic_data 进行扩充
                seismic_data_aug[Cur_aug_key] = [Cur_seismic_data[0][i:i+each_len,:]]
    return labels_aug, seismic_data_aug
def get_aug_data(seismic_data_ts,labels_ts,keys_all):
    """
    :param seismic_data_ts: 目标层段的地震数据
    :param labels_ts:       目标层段的label
    :param keys_all:        [train_key, validation_key, test_key]
    :return: [train_x, train_y],[v_x,v_y], [t_x, t_y]
    """
    seismic_data_aug = []
    labels_aug = []
    for key_cluster in keys_all:
        Cur_seismic_data = {}
        Cur_labels = {}
        for key in key_cluster:
            Cur_seismic_data[key] = seismic_data_ts[key]
            Cur_labels[key] = labels_ts[key]
        # 对train , validation, test 分别进行数据扩充
        Cur_labels_aug, Cur_seismic_data_aug = augment_data(Cur_labels,Cur_seismic_data,each_len=150,interval=5)

        keys_aug = sorted(Cur_labels_aug.keys())
        Cur_seismic_data_ret = [Cur_seismic_data_aug[key] for key in keys_aug]
        Cur_labels_ret = [Cur_labels_aug[key] for key in keys_aug]
        # 需要将y的label对应输入x进行扩展,并将x中元素进行相应变化
        train_x, train_y = change(Cur_seismic_data_ret, Cur_labels_ret)

        seismic_data_aug.append(train_x)
        labels_aug.append(train_y)

    return [seismic_data_aug[0],labels_aug[0]],[seismic_data_aug[1],labels_aug[1]],[seismic_data_aug[2],labels_aug[2]]
def split_data(seismic_data, labels, train=0.6, validation=0.1, test=0.3,data_aug = True,cnnrnn=False):
    """
    首先根据seismic_data 和 labels 的key将其对应起来，然后分别构造训练集，验证集和测试集
    :param seismic_data:    key = well_name, value = [feature matrix]*25
    :param labels:          key = well_name, value = [[bottom,top],labels], bottom 和 top都是时间表示的
    :return:
    """
    print('正在划分数据，data_aug:%s'%(str(data_aug)))
    seismic_data_ts, labels_ts, ts_max = get_ts_data(seismic_data, labels,cnnrnn)  # 获取目标层段的地震数据
    keys = list(seismic_data_ts.keys())
    index = np.random.permutation(len(keys))
    train_key = [keys[i] for i in index[:int(len(keys) * train)].tolist()]
    validation_key = [keys[i] for i in index[int(len(keys) * train):int(len(keys) * (train + validation))].tolist()]
    test_key = [keys[i] for i in index[int(len(keys) * (train + validation)):].tolist()]

    if data_aug:
        # labels_ts, seismic_data_ts = augment_data(labels_ts, seismic_data_ts,each_len=120,interval=2)
        keys_all = [train_key, validation_key, test_key]
        # [seismic_data, label]
        train_data, validation_data, test_data = get_aug_data(seismic_data_ts, labels_ts, keys_all)
        return train_data, validation_data, test_data, ts_max       # 均是扩增后的数据
    else:
        train_x = [seismic_data_ts[key] for key in train_key]
        train_y = [labels_ts[key] for key in train_key]
        validation_x = [seismic_data_ts[key] for key in validation_key]
        validation_y = [labels_ts[key] for key in validation_key]
        test_x = [seismic_data_ts[key] for key in test_key]
        test_y = [labels_ts[key] for key in test_key]

        # 需要将y的label对应输入x进行扩展,并将x中元素进行相应变化
        train_x, train_y = change(train_x, train_y)
        validation_x, validation_y = change(validation_x, validation_y)
        test_x, test_y,test_depth = change(test_x, test_y,return_depth = True)
        return [train_x, train_y], [validation_x, validation_y], [test_x, test_y,test_depth],ts_max


def get_input_shallow(train):
    len_train = len(train[0])
    len_all = 0
    feature_dim = len(train[0][0][0])
    tmp_x = []
    tmp_y = []
    for i in range(len_train):
        len_all += len(train[0][i])
        for j in range(len(train[0][i])):
            tmp_x.append(train[0][i][j])
            tmp_y.append(train[1][i][j])
    train_x = np.array(tmp_x).reshape([len_all, feature_dim])
    train_y = np.array(tmp_y).reshape([len_all])
    return [train_x, train_y]