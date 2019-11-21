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
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()  
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


#tune2
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
'''
#tune3
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
'''
def get_cnn_model(num_classes):
    model = cnn_model()
    # fnn
    model.add(Flatten())
    model.add(Dense(64, activation='relu', name='dense2'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model

class GHMCLoss:
    def __init__(self, bins=10, momentum=0.75):
        self.bins = bins
        self.momentum = momentum
        self.edges_left, self.edges_right = self.get_edges(self.bins)  # edges_left: [bins, 1, 1], edges_right: [bins, 1, 1]
        if momentum > 0:
            self.acc_sum = self.get_acc_sum(self.bins) # [bins]

    def get_edges(self, bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left) # [bins]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-6
        edges_right = tf.constant(edges_right) # [bins]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1, 1]
        return edges_left, edges_right

    def get_acc_sum(self, bins):
        acc_sum = [0.0 for _ in range(bins)]
        return tf.Variable(acc_sum, trainable=False)

    def calc(self, input, target, mask=None, is_mask=False):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        mask [batch_num, class_num]
        """
        edges_left, edges_right = self.edges_left, self.edges_right
        mmt = self.momentum
        # gradient length
        self.g = tf.abs(tf.sigmoid(input) - target) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0) # [1, batch_num, class_num]
        g_greater_equal_edges_left = tf.greater_equal(g, edges_left)# [bins, batch_num, class_num]
        g_less_edges_right = tf.less(g, edges_right)# [bins, batch_num, class_num]
        zero_matrix = tf.cast(tf.zeros_like(g_greater_equal_edges_left), dtype=tf.float32) # [bins, batch_num, class_num]
        if is_mask:
            mask_greater_zero = tf.greater(mask, 0)
            inds = tf.cast(tf.logical_and(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                                          mask_greater_zero), dtype=tf.float32)  # [bins, batch_num, class_num]
            tot = tf.maximum(tf.reduce_sum(tf.cast(mask_greater_zero, dtype=tf.float32)), 1.0)
        else:
            inds = tf.cast(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                           dtype=tf.float32)  # [bins, batch_num, class_num]
            input_shape = tf.shape(input)
            tot = tf.maximum(tf.cast(input_shape[0] * input_shape[1], dtype=tf.float32), 1.0)
        num_in_bin = tf.reduce_sum(inds, axis=[1, 2]) # [bins]
        num_in_bin_greater_zero = tf.greater(num_in_bin, 0) # [bins]
        num_valid_bin = tf.reduce_sum(tf.cast(num_in_bin_greater_zero, dtype=tf.float32))

        # num_in_bin = num_in_bin + 1e-12
        if mmt > 0:
            update = tf.assign(self.acc_sum, tf.where(num_in_bin_greater_zero, mmt * self.acc_sum \
                                  + (1 - mmt) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                self.acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(self.acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = acc_sum + zero_matrix # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)
        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = num_in_bin + zero_matrix # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
        weights = weights / num_valid_bin
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=input)
        loss = tf.reduce_sum(loss * weights) / tot
        return loss

'''

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
    adam2=Adam(lr=0.0005, beta_1=0.9, beta_2=0.99)

    #optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #new_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='mse', metrics=['accuracy'])
#     new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    ghm = GHMCLoss(momentum=0.75)
    new_model.compile(optimizer=adam, loss=ghm, metrics=['accuracy'])

    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_val.shape: ', x_val.shape)
    new_model.fit(x_train, y_train, batch_size=batch_size, epochs=30, shuffle=True,
                  validation_data=(x_val, y_val), callbacks=callbacks_list)
    new_model.load_weights('cnn_weight/Time_cnn_all_gap_slice_0.h5')
    #new_model.save_weights('cnn_weight/zk_weight.h5')
    new_model.save_weights('cnn_weight/zk_weight_total_tune4.h5')
    # test
    score = new_model.evaluate(x_test, y_test, batch_size=256)
    pred = new_model.predict(x_test, batch_size=256)
    #save_file = 'result/cnn_result/Time_cnn_all_gap_slice_0.pkl'
    save_file = 'result/cnn_result/Time_cnn_all_gap_slice_0_total_tune4.pkl'
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
    weight_path = 'cnn_weight/zk_weight_total_tune4.h5'
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

