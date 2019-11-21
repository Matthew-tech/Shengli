# -*- coding:utf-8 -*-
# time:2018/8/6 下午10:03
# author:ZhaoH
from __future__ import division, print_function, absolute_import
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
import pickle as pkl
from data_prepare.data_generator import *
from data_prepare.data_generator_target import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


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


def get_target_cdp(line, cdp, time_s, time_e, sgy_info=[]):
    """
    :param line:
    :param cdp:
    :param time_s:
    :param time_e:
    :param sgy_info: [sgy_dir, sgy filename]
    :return: 指定的纵向预测
    """
    sgyfile = os.path.join(sgy_info[0], sgy_info[1])
    target_data = []
    with open(sgyfile, 'rb') as file:
        for Cur_time in range(int(time_s), int(time_e) + 1):
            Cur_data = get_slice_data(line=line, cdp=cdp, time=Cur_time, seismic_file=file)
            target_data.append(Cur_data)

    print("finished target data!")

    return target_data


def pred_model(num_classes, sgy_info, plane_info):
    """
    :param num_classes: 类别数 2
    :param sgy_info: [sgy路径， sgy文件名]
    :param plane_info: [line, cdp， time1, time2]
    :return: plane_info的预测结果
    """
    # load weight
    weight_path = 'cnn_weight/best_slice_0.h5'
    model = get_cnn_model(num_classes)
    model.load_weights(weight_path)

    # get plane info
    line = plane_info[0]
    cdp = plane_info[1]
    time_s = plane_info[2]
    time_e = plane_info[3]

    # predict
    print("line: {}, cdp: {}, time:{}~{}.".format(line, cdp, time_s, time_e))
    target_data = get_target_cdp(line, cdp, time_s, time_e, sgy_info=sgy_info)
    data_set = np.array(target_data)
    data_set = get_data(data_set, norm='z_score')
    pred_target = model.predict(data_set)
    pred_res = [ele[1] for ele in pred_target]
    file_save = "result/cnn_result/pred_line_{}_cdp_{}.pkl".format(line, cdp)
    with open(file_save, 'wb') as fs:
        pkl.dump(pred_res, fs)
    print("finished predict!")


if __name__ == '__main__':
    # train_model(2)
    sgy_info = ['/disk2/Shengli/data/seismicdata/otherseis/', "petrel_Time_gain_attr.sgy"]
    plane_info = [1191, 1493, 1106.11, 1465.57]
    pred_model(2, sgy_info, plane_info)
    # pred_time(2)