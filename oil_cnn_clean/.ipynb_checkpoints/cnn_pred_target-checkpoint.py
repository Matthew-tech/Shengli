# -*- coding:utf-8 -*-
# time:2018/5/3 上午9:26
# author:ZhaoH
from __future__ import division, print_function, absolute_import
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint, TensorBoard)
import pickle as pkl
from data_prepare.data_generator import *
from data_prepare.data_generator_target import *
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
cdp_s = 1189
cdp_e = 1852
line_s = 627
line_e = 2267


def cnn_model():
    model = Sequential()
    # block1
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv1', input_shape=(14, 14, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # block2
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block2_conv1'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    return model


def get_cnn_model(num_classes):
    model = cnn_model()
    # fnn
    model.add(Flatten())
    model.add(Dense(64, activation='relu', name='dense2'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model


def normalize(data, norm='z_score'):
    """
    对数据进行归一化，data 是要归一化的数据，shape = (28,28,3), norm 是归一化的方式
    :param data:
    :param norm:
    :return:
    """
    norm_paras_file = os.path.join('data_prepare/max_min_mean_std_new.pkl')
    with open(norm_paras_file, 'rb') as file:
        paras = pickle.load(file)  # {filename:[max, min, mean, std] }
        for key in paras.keys():
            if 'petrel_Time_gain_attr' in key:
                Cur_paras = paras.get(key)
                break
    if norm == 'z_score':
        image = (data - Cur_paras[2]) / Cur_paras[3]
    elif norm == 'min_max':
        image = (data - Cur_paras[1]) / (Cur_paras[0] - Cur_paras[1])
    elif norm == 'grey':
        image = np.array((data - Cur_paras[1]) / (Cur_paras[0] - Cur_paras[1]) * 255, dtype=np.int)
    return image


def get_data(data_set, norm='z_score'):
    data_set = np.array(data_set)
    data_new = normalize(data_set, norm)

    return np.array(data_new)


def get_data_batch(data_set, batch_size, norm='z_score'):
    data_set = np.array(data_set)
    data_set = get_data(data_set, norm=norm)
    batch_num = int(len(data_set) / batch_size)
    for batch_i in range(batch_num + 1):
        if batch_i == batch_num:
            x_batch = data_set[batch_i * batch_size:]
        else:
            start_i = batch_i * batch_size
            x_batch = data_set[start_i:start_i + batch_size]

        yield x_batch


def pred_model(num_classes, sgy_info, plane_info):
    """
    :param num_classes: 类别数 2
    :param sgy_info: [sgy路径， sgy文件名]
    :param plane_info: [顶底文件路径, t0文件， t1文件]
    :return: result/cnn_target/pred_line_{}to{}.pkl 按照line存储的目标层段预测数据
    """
    # load weight
    weight_path = 'cnn_weight/best_slice_0.h5'
    model = get_cnn_model(num_classes)
    model.load_weights(weight_path)

    # preprocess plane_dir
    file_plane_path = plane_info[0]  # "data/plane_loc/"
    plane_t0 = plane_info[1]  # "t0_grid_28jun_155317.p701"
    plane_t11 = plane_info[2]  # "t11_grid_28jun_155411.p701"
    this_file_s = os.path.join(file_plane_path, plane_t0)  # 横向切面的文件_s
    this_file_e = os.path.join(file_plane_path, plane_t11)  # 横向切面的文件_e
    assert os.path.exists(this_file_s), this_file_s + 'no'
    assert os.path.exists(this_file_e), this_file_e + 'no'
    plane_file_pkl_s = "{}.pkl".format(plane_t0)
    plane_file_pkl_save_s = os.path.join(file_plane_path, plane_file_pkl_s)
    plane_file_pkl_e = "{}.pkl".format(plane_t11)
    plane_file_pkl_save_e = os.path.join(file_plane_path, plane_file_pkl_e)
    # 将切面文件的信息保存下来
    if not os.path.exists(plane_file_pkl_save_s):
        print('generating: %s' % plane_file_pkl_save_s)
        with open(this_file_s, 'r') as file1, open(plane_file_pkl_save_s, 'wb') as file2:
            line = file1.readline()
            """ loc_time_dict_s """
            """
                修改了！！！！！
            """
            loc_time_dict_s = {}
            while line and line != []:
                line_split = line[:-1].split(' ')
                line_split = [value for loc, value in enumerate(line_split) if value != '']
                line_num = line_split[0]
                cdp_num = str(int(float(line_split[1])))
                time = line_split[4]
                loc_time_dict_s[line_num + '-' + cdp_num] = float(time)
                line = file1.readline()
            # print(loc_time_dict)
            pickle.dump(loc_time_dict_s, file2, -1)
    with open(plane_file_pkl_save_s, 'rb') as file:
        print('loading...%s' % plane_file_pkl_save_s)
        loc_time_dict_s = pickle.load(file)  # key:str line-cdp, value:float(time)
        print(len(loc_time_dict_s))
        # if len(loc_time_dict) != 1068468: return
    if not os.path.exists(plane_file_pkl_save_e):
        print('generating: %s' % plane_file_pkl_save_e)
        with open(this_file_e, 'r') as file1, open(plane_file_pkl_save_e, 'wb') as file2:
            line = file1.readline()
            """ loc_time_dict_e """
            loc_time_dict_e = {}
            while line and line != []:
                line_split = line[:-1].split(' ')
                line_split = [value for loc, value in enumerate(line_split) if value != '']
                line_num = line_split[0] # 4
                cdp_num = str(int(float(line_split[1]))) # 3
                time = line_split[4] # 2
                loc_time_dict_e[line_num + '-' + cdp_num] = float(time)
                line = file1.readline()
            # print(loc_time_dict)
            pickle.dump(loc_time_dict_e, file2, -1)
    with open(plane_file_pkl_save_e, 'rb') as file:
        print('loading...%s' % plane_file_pkl_save_e)
        loc_time_dict_e = pickle.load(file)  # key:str line-cdp, value:float(time)
        print(len(loc_time_dict_e))
        #exit()
        # if len(loc_time_dict) != 1068468: return
    # file_save = 'result/cnn_result/Time_pred_slice_1_52.pkl'

    # 按照line进行循环
    for cur_line in range(line_s + 2, line_e - 25 - 4 + 1, 10):
        print("cur_line: ", cur_line)
        file_save = "result/cnn_target/pred_line_{}to{}.pkl".format(cur_line, cur_line+9)
        if os.path.exists(file_save):
            continue
        cur_line_s = cur_line
        cur_line_e = cur_line+9
        if cur_line_e > line_e - 25:
            cur_line_e = line_e - 25
        cur_ten_line_data = get_target_horizon_slice(cur_line_s, cur_line_e,
                                                     loc_time_dict_s, loc_time_dict_e,
                                                     sgy_info=sgy_info)
        if cur_ten_line_data == []:
            continue
        else:
            if not os.path.exists(file_save):
                print('initializing...')
                batch_size = 10240
                data_batch = get_data_batch(cur_ten_line_data, batch_size, norm='z_score')
                pred_res = []
                i_t = 0
                for batch_x in data_batch:
                    i_t += 1
                    pred = model.predict(batch_x, batch_size=batch_size)
                    pred_new = [ele[1] for ele in pred]
                    pred_res.append(pred_new)
                    cur_rate = i_t * 100 / (len(cur_ten_line_data) // batch_size + 1)
                    print('cur_rate: ', cur_rate, '%')
                with open(file_save, 'wb') as fs:
                    pkl.dump(pred_res, fs)


if __name__ == '__main__':
    # train_model(2)
    sgy_info = ['/disk2/Shengli/data/seismicdata/otherseis/', "petrel_Time_gain_attr.sgy"]
    plane_info = ["data/plane_loc/", "t1int.txt", "tr.txt"]
    pred_model(2, sgy_info, plane_info)
    # pred_time(2)