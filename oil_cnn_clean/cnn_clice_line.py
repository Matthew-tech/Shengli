# -*- coding:utf-8 -*-
# time:2018/5/3 上午9:26
# author:ZhaoH
from __future__ import division, print_function, absolute_import
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
import pickle as pkl
from data_prepare.data_generator import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
'''

os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
session = tf.Session(config = config)


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


def turn_into_specific_channel(image, method, norm='z_score'):
    image = normalize(image, norm=norm)
    if method == 3:
        return image
    elif method == 1:
        temp = np.zeros(shape=(14 * 2, 14 * 2))
        temp[:14, :14] = image[:, :, 0]
        temp[:14, 14:] = image[:, :, 1]
        temp[14:, :14] = image[:, :, 2]
        return temp.reshape((14 * 2, 14 * 2, 1))
    elif method == 2:
        '''
        temp = np.zeros(shape=(14, 14))
        temp[:14, :14] = image[:, :, 0]
        return temp.reshape((14, 14, 1))
        '''
        temp = np.zeros(shape=(14, 14))
        temp[:14, :14] = image[:, :, 1]
        return temp.reshape((14, 14, 1))

def get_data(data_set, method, norm='z_score'):
    data_set = np.array(data_set)
    data_new = []
    count = 0
    for cur_image in data_set:
        count += 1
        print('get_data_rate: ', float(count / len(data_set)))
        tmp_image = turn_into_specific_channel(cur_image, method, norm=norm)
        data_new.append(tmp_image)

    return np.array(data_new)


def get_data_batch(data_set, batch_size, method, norm='z_score'):#返回这个batch在数据集中的范围
    data_set = np.array(data_set)
    data_set = get_data(data_set, method, norm=norm)
    batch_num = int(len(data_set) / batch_size)
    for batch_i in range(batch_num + 1):
        if batch_i == batch_num:
            x_batch = data_set[batch_i * batch_size:]
        else:
            start_i = batch_i * batch_size
            x_batch = data_set[start_i:start_i + batch_size]

        yield x_batch


def pred_model(num_classes, file_plane, save_file):
    """
    :param num_classes: 类别数 2
    :param file_plane: 提取后的平面数据路径
    :param save_file: 保存的平面预测结果路径（.pkl格式）
    :return: 生成平面预测结果（.pkl格式）
    """
    # load weight
    weight_path = 'cnn_weight/best_slice_0.h5'
    model = get_cnn_model(num_classes)
    model.load_weights(weight_path)

    # get input
    file_dir = file_plane  # 'data/cnn_test/petrel_Time_gain_attr.sgy_ngs52_grid_28jun_155214.p701.ht'
    file_save = save_file  # 'result/cnn_result/Time_pred_slice_1_52.pkl'

    with open(file_dir, 'rb') as file1:
        data_set = pkl.load(file1)

    print('initializing...')
    batch_size = 10240
    data_batch = get_data_batch(data_set, batch_size, 2, norm='z_score')
    pred_res = []
    i_t = 0
    for batch_x in data_batch:
        i_t += 1
        pred = model.predict(batch_x, batch_size=batch_size)
        pred_new = [ele[1] for ele in pred]
        pred_res.append(pred_new)
        cur_rate = i_t * 100 / (len(data_set) // batch_size + 1)
        #print('打印预测结果(pred):\n',pred)
        #print('打印预测结果(pred_new):\n',pred_new)
        print('cur_rate: ', cur_rate, '%')
    #print('打印result/cnn_result/Time_pred_slice_1_52.pkl中存储的信息:',pred_res)
    print('length of pred_res',len(pred_res))
    with open(file_save, 'wb') as fs:
        pkl.dump(pred_res, fs)


if __name__ == '__main__':
    file_plane = '../oil_cnn/data/cnn_test/petrel_Time_gain_attr.sgy_ngs52_grid_28jun_155214.p701.ht'
    #file_plane = 'data/cnn_test/petrel_Time_gain_attr.sgy_ngs52_grid_28jun_155214.p701.ht'
    save_file = 'result/cnn_result/Time_pred_slice_1_52.pkl'
    pred_model(2, file_plane, save_file)