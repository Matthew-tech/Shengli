import os
from data_prepare.select_high_correlation_attrs import check_is_high_correlation
import struct
import numpy as np
import time
import pickle
from Configure.global_config import *
import shutil

sampling_points = 1251


def get_files():
    cube_dir = file_loc_gl.seismic_sgy_file_path_base
    file_list = []
    for child_dir in os.listdir(cube_dir):
        for file in os.listdir(os.path.join(cube_dir, child_dir)):
            if not check_is_high_correlation(child_dir, child_dir + '-' + file)[0]:
                continue
            file_list.append([file, os.path.join(cube_dir, child_dir)])
    return sorted(file_list, key=lambda x: x[0])


def get_each_trace_data(file):
    file.read(240)
    Cur_trace_data = []
    for point_i in range(sampling_points):
        Cur_trace_data.append(struct.unpack('!f', file.read(4))[0])
    return Cur_trace_data


def get_min_max(file_list):
    """
    :param file_list: get_files()
    :return: min_data, max_data
    """
    min_data = float('inf')
    max_data = float('-inf')
    for filepath in file_list:
        with open(os.path.join(filepath[1], filepath[0]), 'rb') as file:
            # print(filepath[0])
            file.read(3600)  # 跳过道头
            cur_trace = get_each_trace_data(file)
            while cur_trace is not None:
                cur_trace = np.array(cur_trace)
                min_cur = np.min(cur_trace)
                max_cur = np.max(cur_trace)
                if min_cur < min_data:
                    min_data = min_cur
                if max_cur > max_data:
                    max_data = max_cur
                # print('min: ', min_data)
                cur_trace = get_each_trace_data(file)
        print(filepath[0])
        print('min: ', min_data)
        print('max: ', max_data)


def get_all_trace_data(file, filepath):
    file.read(3600)  # 跳过道头
    full_data = []
    Cur_trace_data = get_each_trace_data(file)
    while Cur_trace_data is not None:
        full_data.append(Cur_trace_data)
        if os.path.getsize(filepath) == file.tell():
            break
        Cur_trace_data = get_each_trace_data(file)
    return full_data


def get_min_max_mean_std(file_list):
    """
    :param file_list: get_files()
    :return:
    """
    dict_all = {}
    count = 1
    for filepath in file_list:
        full_data = []
        full_data_np = []
        with open(os.path.join(filepath[1], filepath[0]), 'rb') as file:
            print(filepath[0])
            print('count:', count)
            count += 1
            full_data = get_all_trace_data(file, os.path.join(filepath[1], filepath[0]))
            full_data_np = np.array(full_data).reshape([len(full_data) * sampling_points])
            min_data = np.min(full_data_np)
            print('min_data: ')
            max_data = np.max(full_data_np)
            print('max_data: ')
            mean_data = np.mean(full_data_np)
            print('mean_data: ')
            std_data = np.std(full_data_np)
            print('std_data: ')
            dict_tmp = {filepath[0]: [max_data, min_data, mean_data, std_data]}
            dict_all = dict(dict_all, **dict_tmp)
            #print(dict_all)
    with open('Results/max_min_mean_std_new.pkl', 'wb') as writefile:
        pickle.dump(dict_all, writefile)
    shutil.copy('Results/max_min_mean_std_new.pkl', 'data/full_train_data/max_min_mean_std_new.pkl')


def get_new():
    file_path = 'Results/max_min_mean_std.pkl'
    save_path = 'Results/max_min_mean_std_new.pkl'
    cube_dir = file_loc_gl.seismic_sgy_file_path_base
    with open(file_path, 'rb') as readfile:
        a = pickle.load(readfile)
        print(a)
        count = 1
        for child_dir in os.listdir(cube_dir):
            for file in os.listdir(os.path.join(cube_dir, child_dir)):
                if not check_is_high_correlation(child_dir, child_dir + '-' + file)[0]:
                    continue
                data_r = pickle.load(readfile)
                file_ = {file}
                print(data_r.keys())
                print(file_)
                while data_r.keys() != file_:
                    data_r = pickle.load(readfile)
                with open(save_path, 'wb') as writefile:
                    pickle.dump(data_r, writefile)
                print('key: ', data_r.keys(), count)
                count += 1


def readpkl():
    file_path = 'Results/max_min_mean_std_new.pkl'
    with open(file_path, 'rb') as readfile:
        data_dict = pickle.load(readfile)
        print(data_dict.keys())


if __name__ == '__main__':
    #file_list = get_files()
    #get_min_max_mean_std(file_list)
    ## get_new()
    readpkl()
