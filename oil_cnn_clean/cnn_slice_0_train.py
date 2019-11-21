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
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import *
from keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=0
config = tf.ConfigProto()  
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


def cnn_model():
    model = Sequential()
    # block1
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv1', input_shape=(14, 14, 1)))

    model.add(Conv2D(128, (1, 1), padding='same', activation='relu', name='block1_conv2'))
    #model.add(Conv2D(128, (1, 1), padding='same', activation='relu', name='block1_conv3'))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu', name='block1_conv4'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv5'))
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # block2
    #for i in range(5):
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))#, name='block2_conv1'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block2_conv2'))
    #model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block2_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='block2_conv4'))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='block2_conv5'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='block3_conv1'))
    #model.add(Conv2D(128, (1, 1), padding='same', activation='relu', name='block3_conv1'))
    #model.add(GlobalAveragePooling2D())
    #block3
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='block3_conv1'))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='block3_conv2'))
    # model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='block3_conv3'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #block4
    #model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block4_conv1'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    return model


def get_cnn_model(num_classes):
    model = cnn_model()
    # fnn
    model.add(Flatten())
    
    #model.add(Dense(256, activation='relu', name='dense1'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu',  name='dense2'))
    model.add(Dropout(0.1))
    #model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
    #            activity_regularizer=regularizers.l1(0.01), name='dense3'))
    #model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model
'''
def cnn_model():
    model = Sequential()
    # block1
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1', input_shape=(14, 14, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv5', input_shape=(14, 14, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv6', input_shape=(14, 14, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv7', input_shape=(14, 14, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # block2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv1'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv2'))
    #model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    #model.add(GlobalAveragePooling2D())

    return model


def get_cnn_model(num_classes):
    model = cnn_model()
    # fnn
    model.add(Flatten())
    
    #model.add(Dense(256, activation='relu', name='dense1'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='dense2'))
    model.add(Dropout(0.1))
    #model.add(Dense(128, activation='relu', name='dense3'))
    #model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model
'''
def train_model(num_classes):
    # get input
    filepath = 'data/cnn_train/'#由data——generation.py生成
    #batch_size = 32   256good
    batch_size = 256
    num_classes = num_classes
    image_reader = ImageReader(filepath, batch_size=batch_size,
                               partition=[0.6, 0.2, 0.2], method=2,
                               norm='z_score')
    train_dict = image_reader.train_batch()
    test_dict = image_reader.test_batch()
    val_dict = image_reader.val_batch()
    x_train = np.array(train_dict['image'])
    y_train = np.array(train_dict['label'])
    x_val = np.array(val_dict['image'])
    y_val = np.array(val_dict['label'])
    x_test = np.array(test_dict['image'])
    y_test = np.array(test_dict['label'])

    # load weight
    new_model = get_cnn_model(num_classes)
    checkpoint = ModelCheckpoint('cnn_weight/Time_cnn_all_gap_slice_0.h5', monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    tb = TensorBoard(log_dir='graph/Time_cnn_all_gap_slice_0', write_graph=True)
    callbacks_list = [checkpoint, early, tb]
    
    # train
    adam = Adam(lr=0.0001, decay=1e-6)
    #optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #new_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='mse', metrics=['accuracy'])
    new_model.compile(optimizer=Adam(lr=0.0005, beta_1=0.9,beta_2=0.99), loss='categorical_crossentropy', metrics=['accuracy'])

    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    new_model.fit(x_train, y_train, batch_size=batch_size, epochs=30, shuffle=True,
                  validation_data=(x_val, y_val), callbacks=callbacks_list)
    new_model.load_weights('cnn_weight/Time_cnn_all_gap_slice_0.h5')
    #new_model.save_weights('cnn_weight/zk_weight.h5')
    new_model.save_weights('cnn_weight/zk_weight_total.h5')
    # test
    score = new_model.evaluate(x_test, y_test, batch_size=256)
    pred = new_model.predict(x_test, batch_size=256)
    #save_file = 'result/cnn_result/Time_cnn_all_gap_slice_0.pkl'
    save_file = 'result/cnn_result/Time_cnn_all_gap_slice_0_total.pkl'
    with open(save_file, 'wb') as sf:
        pkl.dump({'pred': pred, 'label': y_test}, sf)
    print(pred.shape)
    print(y_test.shape)

    print('test loss:', score[0])
    print('test acc:', score[1])


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
        #time切片

        temp = np.zeros(shape=(14, 14))
        temp[:14, :14] = image[:, :, 0]
        return temp.reshape((14, 14, 1))  

        '''
        #line切片
        temp = np.zeros(shape=(14, 14))
        temp[:14, :14] = image[:, :, 1]
        return temp.reshape((14, 14, 1))

        #cdp切片
        temp = np.zeros(shape=(14, 14))
        temp[:14, :14] = image[:, :, 2]
        return temp.reshape((14, 14, 1))
        '''
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


def get_data_batch(data_set, batch_size, method, norm='z_score'):
    data_set = np.array(data_set)
    data_set = get_data(data_set, method, norm=norm)
    batch_num = int(len(data_set) / batch_size)
    for batch_i in range(batch_num+1):
        if batch_i == batch_num:
            x_batch = data_set[batch_i*batch_size:]
        else:
            start_i = batch_i * batch_size
            x_batch = data_set[start_i:start_i + batch_size]

        yield x_batch


def pred_model(num_classes):
    # load weight
    #weight_path = 'cnn_weight/zk_weight.h5'
    weight_path = 'cnn_weight/zk_weight_total.h5'
    model = get_cnn_model(num_classes)
    model.load_weights(weight_path)

    # get input
    #file_dir = 'data/cnn_test/petrel_Time_gain_attr.sgy_ngs52_grid_28jun_155214.p701.ht'
    #file_save = 'result/cnn_result/Time_pred_slice_1_52.pkl'
    file_dir = 'data/cnn_test/petrel_Time_gain_attr.sgy_ng33sz_grid_28jun_154331.p701.ht'
    # file_save = 'result/cnn_result/Time_pred_gap_slice_0_33.pkl'
    file_save = 'result/cnn_result/Time_pred_gap_slice_0_33_total.pkl'
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
        print('cur_rate: ', cur_rate, '%')
    with open(file_save, 'wb') as fs:
        pkl.dump(pred_res, fs)
    

def pred_time(num_classes):
    # load weight
    weight_path = 'cnn_weight/best_slice_1.h5'
    model = get_cnn_model(num_classes)
    model.load_weights(weight_path)

    # loop
    horizon_data_dir = '../cnn_test'
    for h_i in range(-9, 9):
        horizon_data_filename = "petrel_Time_gain_attr.sgy_ng33sz_grid_28jun_154331.p701_{}.ht".format(h_i)
        print(horizon_data_filename)
        cur_file_dir = os.path.join(horizon_data_dir, horizon_data_filename)

        # get input
        file_save = 'result/time_result/pred_ng33_{}.pkl'.format(h_i)

        with open(cur_file_dir, 'rb') as file1:
            data_set = pkl.load(file1)

        print('initializing...')
        batch_size = 2048
        data_batch = get_data_batch(data_set, batch_size, 2, norm='z_score')
        pred_res = []
        i_t = 0
        for batch_x in data_batch:
            i_t += 1
            pred = model.predict(batch_x, batch_size=batch_size)
            pred_new = [ele[1] for ele in pred]
            pred_res.append(pred_new)
            cur_rate = i_t * 100 / (len(data_set) // batch_size + 1)
            print('cur_rate: ', cur_rate, '%')
        with open(file_save, 'wb') as fs:
            pkl.dump(pred_res, fs)
    
    
if __name__ == '__main__':
    train_model(2)
    #pred_model(2)#zk
    #train_model(2)#zkd
    #pred_time(2)#hhhhhhzk

