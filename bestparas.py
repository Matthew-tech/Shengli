#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-28 下午11:10
# @Author  : zejin
# @File    : bestparas.py
import shutil
import pickle as pkl
import os
from Configure.global_config import file_loc_gl

def best_paras_pkl():
    paras = {}
    paras['count'] = 0
    paras['model'] = 'BiRNN'
    paras['layers'] = 2
    paras['trace_range'] = 0
    paras['normalize'] = 'MN'
    paras['cellsize'] = 16
    paras['dropout'] = 0.3
    paras['rnn_cell'] = 'GRU'
    paras['use_alllabel'] = False
    paras['opt'] = 'Adam'
    paras['loss'] = 'CEE'

    # 保存最优参数组合
    paras_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/best_paras')
    if os.path.isdir(paras_dir):
        shutil.rmtree(paras_dir)
        os.mkdir(paras_dir)
    else:
        os.mkdir(paras_dir)
    with open(os.path.join(paras_dir, 'best_paras.pkl'), 'wb') as f:
        pkl.dump(paras, f)


if __name__ == '__main__':
    #model_evaluation()
    best_paras_pkl()
    print("model training finish!")