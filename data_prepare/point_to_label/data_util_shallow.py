#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-7 下午7:23
# @Author  : Eric
# @File    : data_util_shallow.py
# 生成浅层方法用到的数据输入
import sys
import time
sys.path.append("/disk3/zk/aboutoil/Shengli")
from Configure.global_config import file_loc_gl
import os
from data_prepare.point_to_label.get_input_data_p2l import get_labels, get_seismic_data, get_ts_data
import pickle
import numpy as np
import pandas as pd
def merge(seismic_data,labels):
    """
    将 seismic_data 和 labels合并到一起
    :param seismic_data: map，key： well_name, value: list
    :param labels:       map,key : well_name, value: list
    :return:
    """
    data_ret = {}
    for key in seismic_data.keys():

        Cur_seismic_data = seismic_data[key][0]     # shape = (None,76)
        ts_len = len(Cur_seismic_data)
        Cur_label = np.asarray(labels[key][1][:ts_len]).T.reshape([-1,1])
        # 将label加入最后一列
        Cur_data = np.hstack((Cur_seismic_data, Cur_label))
        data_ret[key] = Cur_data
    return data_ret
def save_cur_samples(normalize='GN',target_seg = True):
    """
    将单个样本保存下来，分别将其做高斯和线性归一化并保存,保存位置为4-training_data/shallow_methods/samples_%s(normalize)
    :param normalize: 样本的归一化方式
    :param target_seg: True表示只保存目标层段，False表示保存所有的标记值
    :return:
    """
    des_dir = os.path.join(file_loc_gl.training_data_dir,'shallow_methods')
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    all_labels = get_labels(use_alllabels=not target_seg)  # use_alllabel = True 表示使用全部的标记数据，False表示只使用目标层段
    all_seismic_data = get_seismic_data(trace_range=0, normalize=normalize)
    # 得到目标层段的seismic_data 和 labels
    seismic_data_ts, labels_ts, ts_max = get_ts_data(all_seismic_data, all_labels)  # 获取目标层段的地震数据
    # 将所有的数据变成特征+1列label列,然后将其保存到本地
    seismic_labels_dict = merge(seismic_data_ts, labels_ts)
    des_file = os.path.join(des_dir,'samples_%s_ts_%s.pkl'%(normalize, str(target_seg)))
    with open(des_file,'wb') as file:
        pickle.dump(file=file,obj=seismic_labels_dict)

def save_samples():
    for normalize in ['GN','MN']:
        for target_seg in [True, False]:
            des_dir = os.path.join(file_loc_gl.training_data_dir, 'shallow_methods')
            des_file = os.path.join(des_dir, 'samples_%s_ts_%s.pkl' % (normalize, str(target_seg)))
            if os.path.exists(des_file): continue
            save_cur_samples(normalize=normalize,target_seg = target_seg)

def get_part_data(all_data, keys,return_samples = False):
    """
    获取训练数据和label
    :param all_data: 所有的数据，dict，key：well_name ，value: feature + 一列 label
    :param keys:    well_name
    :param return_samples False 不返回以井为单位的数据
    :return:  特征和label
    """
    X = None
    if return_samples:
        samples = {}
    for key in keys:
        if return_samples:
            samples[key] = [all_data[key][:,:-1],all_data[key][:,-1]]
        if X is None:
            X = all_data[key]
        else:
            X = np.vstack((X,all_data[key]))
    if return_samples:
        return X[:,:-1], X[:,-1], samples
    else:
        return X[:,:-1], X[:,-1]
def get_input(paras = {'norm':'GN','ts':True},part = {'train':0.6, 'validation':0.2,'test':0.2}):
    """
    获取浅层方法的输入
    :param paras: 表示需要使用的文件参数
    :param part:  表示训练集，验证集，测试集的划分比例
    :return: 训练集，验证集，测试集       [X, y]
    """
    source_dir = os.path.join(file_loc_gl.training_data_dir,'shallow_methods')
    source_filename = 'samples_%s_ts_%s.pkl'%(paras['norm'], str(paras['ts']))
    with open(os.path.join(source_dir,source_filename),'rb') as file:
        data = pickle.load(file)
        keys = list(data.keys())
        index = np.random.permutation(len(keys))
        train_key = [keys[i] for i in index[:int(len(keys) * part['train'])].tolist()]
        validation_key = [keys[i] for i in index[int(len(keys) * part['train']):
                                                 int(len(keys) * (part['train'] + part['validation']))].tolist()]
        test_key = [keys[i] for i in index[int(len(keys) * (part['train'] + part['validation'])):].tolist()]
        train_data = get_part_data(data,train_key)
        validation_data = get_part_data(data,validation_key)
        test_data = get_part_data(data, test_key,return_samples=True)
    return train_data, validation_data, test_data
if __name__ == '__main__':
    # save_samples()#原文件__main__只有这一行

    train_data, validation_data, test_data=get_input(paras = {'norm':'GN','ts':False})
    pd.to_pickle(train_data,"/disk3/zk/aboutoil/Shengli/data/4-training_data/shallow_methods/xgboost_train_data_76d.pkl")
    pd.to_pickle(validation_data,"/disk3/zk/aboutoil/Shengli/data/4-training_data/shallow_methods/xgboost_val_data_76d.pkl")
    pd.to_pickle(test_data,"/disk3/zk/aboutoil/Shengli/data/4-training_data/shallow_methods/xgboost_test_data_76d.pkl") 