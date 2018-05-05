# -*- coding:utf-8 -*-
# time:2018/4/11 上午10:24
# author:ZhaoH
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt


class ImageReader(object):
    def __init__(self, data_dir, feature_list, input_size, batch_size, partition, method=3, norm='min_max'):
        """
        
        :param data_dir: 
        :param feature_list: must n^2
        :param input_size: [0, 14]
        :param batch_size: 
        :param partition: 
        :param method:  1、单属性单通道
                        2、多属性单通道
                        3、单属性3通道
        :param norm: 
        """
        print('initializing...')
        self._trace_step = 14
        self._data_dir = data_dir
        self._feature_list = feature_list
        self._input_size = input_size
        self._batch_size = batch_size
        self._partition = partition
        self._method = method
        self._norm = norm
        self._train = self._partition[0]
        self._val = self._partition[1]
        self._test = self._partition[2]
        self._feature_num = len(feature_list)
        # self._pos_filename = []
        # self._neg_filename = []
        # self._data_pos = []
        # self._data_neg = []
        for filename in os.listdir(self._data_dir):
            if self._feature_list[0] in filename:
                if 'pos.td' in filename:
                    self._pos_filename = os.path.join(self._data_dir,filename)
                elif 'neg.td' in filename:
                    self._neg_filename = os.path.join(self._data_dir,filename)
        with open(self._pos_filename, 'rb') as file1, open(self._neg_filename, 'rb') as file2:
            self._data_pos = pickle.load(file1)
            self._data_neg = pickle.load(file2)
        self._wells = list(set(list(self._data_pos.keys())+list(self._data_neg.keys())))
        random_index = list(range(len(self._wells)))
        random.shuffle(random_index)
        self._wells = [self._wells[i] for i in random_index]
        self._wells_train = self._wells[:int(len(self._wells)*self._train)]
        self._wells_val = self._wells[int(len(self._wells) * self._train):int(len(self._wells) * (self._train+self._val))]
        self._wells_test = self._wells[int(len(self._wells) * (self._train+self._val)):]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._image_list_train, self._label_list_train = self.get_image_data(self._feature_list[0], type_='train', norm=self._norm)
        self._image_list_val, self._label_list_val = self.get_image_data(self._feature_list[0], type_='val', norm=self._norm)
        self._image_list_test, self._label_list_test = self.get_image_data(self._feature_list[0], type_='test', norm=self._norm)
        self._sample_sum_train = len(self._image_list_train)
        self._sample_sum_val = len(self._image_list_val)
        self._sample_sum_test = len(self._image_list_test)

        self._merge_image_train = np.ones(shape=(self.sample_num_train,
                                                  int(np.sqrt(self._feature_num))*self._input_size,
                                                  int(np.sqrt(self._feature_num))*self._input_size,
                                                  1))*1000
        self._merge_image_val = np.ones(shape=(self.sample_num_val,
                                                  int(np.sqrt(self._feature_num))*self._input_size,
                                                  int(np.sqrt(self._feature_num))*self._input_size,
                                                  1))*1000
        self._merge_image_test = np.ones(shape=(self.sample_num_test,
                                                  int(np.sqrt(self._feature_num))*self._input_size,
                                                  int(np.sqrt(self._feature_num))*self._input_size,
                                                  1))*1000
        self._merge_image_train = self.merge_image(self._merge_image_train, self._image_list_train)
        self._merge_image_test = self.merge_image(self._merge_image_test, self._image_list_test)
        self._merge_image_val = self.merge_image(self._merge_image_val, self._image_list_val)
        for feature in self._feature_list[1:]:
            # i = 0
            for filename in os.listdir(self._data_dir):
                if feature in filename:
                    if 'pos.td' in filename:
                        self._pos_filename = os.path.join(self._data_dir,filename)
                    elif 'neg.td' in filename:
                        self._neg_filename = os.path.join(self._data_dir,filename)
            # i += 1

        # for i in range(self._feature_num):
            with open(self._pos_filename, 'rb') as file1, open(self._neg_filename, 'rb') as file2:
                self._data_pos = pickle.load(file1)
                self._data_neg = pickle.load(file2)
            self._wells = list(set(list(self._data_pos.keys())+list(self._data_neg.keys())))
            # random_index = list(range(len(self._wells)))
            # random.shuffle(random_index)
            self._wells = [self._wells[i] for i in random_index]
            self._wells_train = self._wells[:int(len(self._wells)*self._train)]
            self._wells_val = self._wells[int(len(self._wells) * self._train):int(len(self._wells) * (self._train+self._val))]
            self._wells_test = self._wells[int(len(self._wells) * (self._train+self._val)):]
            self._epochs_completed = 0
            self._index_in_epoch = 0
            self._image_list_train, self._label_list_train = self.get_image_data(feature, type_='train', norm=self._norm)
            self._image_list_val, self._label_list_val = self.get_image_data(feature, type_='val', norm=self._norm)
            self._image_list_test, self._label_list_test = self.get_image_data(feature, type_='test', norm=self._norm)

            self._merge_image_train = self.merge_image(self._merge_image_train, self._image_list_train)
            self._merge_image_test = self.merge_image(self._merge_image_test, self._image_list_test)
            self._merge_image_val = self.merge_image(self._merge_image_val, self._image_list_val)

    def merge_image(self, merge_image, cur_image):
        cur_image = np.array(cur_image).reshape(([len(cur_image), self._input_size, self._input_size, 1]))
        flag = False
        for ele_x in range(len(merge_image[0])):
            for ele_y in range(len(merge_image[0][0])):
                if merge_image[0][ele_x][ele_y][0] == 1000:
                    merge_image[:, ele_x:ele_x+self._input_size, ele_y:ele_y+self._input_size, :] = cur_image
                    flag = True
                    break
            if flag:
                break

        return merge_image

    def normalize(self, data, feature, norm='z_score'):
        """
        对数据进行归一化，data 是要归一化的数据，shape = (28,28,3), norm 是归一化的方式
        :param data:
        :param norm:
        :return:
        """
        norm_paras_file = os.path.join('../', '../Results/max_min_mean_std_new.pkl')
        with open(norm_paras_file, 'rb') as file:
            paras = pickle.load(file)  # {filename:[max, min, mean, std] }
            for key in paras.keys():
                if feature in key:
                    Cur_paras = paras.get(key)
                    break
        if norm == 'z_score':
            image = (data - Cur_paras[2])/Cur_paras[3]
        elif norm == 'min_max':
            image = (data - Cur_paras[1])/(Cur_paras[0]-Cur_paras[1])
        elif norm == 'grey':
            image = np.array((data - Cur_paras[1])/(Cur_paras[0]-Cur_paras[1])*255, dtype=np.int)
        return image

    def turn_into_specific_channel(self, image, feature, norm='z_score'):
        image = self.normalize(image, feature, norm=norm)
        if self._method == 3:
            return image
        elif self._method == 1:
            temp = np.zeros(shape=(self._trace_step*2, self._trace_step*2))
            temp[:self._trace_step, :self._trace_step] = image[:, :, 0]
            temp[:self._trace_step, self._trace_step:] = image[:, :, 1]
            temp[self._trace_step:, :self._trace_step] = image[:, :, 2]
            return temp.reshape((self._trace_step*2,self._trace_step*2,1))
        elif self._method == 2:
            temp = np.zeros(shape=(self._input_size, self._input_size))
            start = (int(self._trace_step/2)+1-int(self._input_size/2))
            temp[:self._input_size, :self._input_size] = image[start:start+self._input_size,
                                                        start:start+self._input_size, 0]
            return temp.reshape((self._input_size,self._input_size,1))

    def get_image_data(self, feature, type_='train', norm='z_score'):
        """
        获取 train，val或test data
        :param type_:
        :param norm: z_score 表示进行高斯归一化，min_max 表示进行线性归一化
        :return:
        """
        if type_ == 'train':
            Cur_wells = self._wells_train
        elif type_ == 'val':
            Cur_wells = self._wells_val
        elif type_ == 'test':
            Cur_wells = self._wells_test
        images_list = []
        labels_list = []
        # 将正样本放进去
        for key in self._data_pos.keys():
            if key in Cur_wells:
                for image_info in self._data_pos.get(key):
                    images_list.append(self.turn_into_specific_channel(list(image_info.values())[0], feature, norm=norm))
                    labels_list.append(1)
        # 将负样本放进去
        for key in self._data_neg.keys():
            if key in Cur_wells:
                for image_info in self._data_neg.get(key):
                    images_list.append(self.turn_into_specific_channel(list(image_info.values())[0], feature, norm=norm))
                    labels_list.append(0)

        # random_index = list(range(len(images_list)))
        # random.shuffle(random_index)
        # images_list = [images_list[i] for i in random_index]
        # labels_list = [labels_list[i] for i in random_index]
        return images_list, labels_list

    @property
    def wells_train(self): return self._train

    @property
    def wells_val(self): return self._val

    @property
    def wells_test(self): return self._val

    @property
    def method(self): return self._method

    @property
    def sample_num_train(self): return self._sample_sum_train

    @property
    def sample_num_val(self): return self._sample_sum_val

    @property
    def sample_num_test(self): return self._sample_sum_test

    def get_all_data(self,shuffle=True):
        # 得到所有的训练数据
        pass

    def one_hot(self,labels):
        #print(labels)
        for i,sample in enumerate(labels):
            labels[i] = [0,1] if sample == 1 else [1,0]
        #print('after:',labels)
        return labels

    def next_batch(self):
        #
        start = self._index_in_epoch
        self._index_in_epoch += self._batch_size
        if self._index_in_epoch > self._sample_sum_train:
            # finish epoch
            self._epochs_completed += 1

            # shuffle the data
            data_index = list(range(self._sample_sum_train))
            random.shuffle(data_index)

            # 将样本打乱
            if self._method != 2:
                self._image_list_train = [self._image_list_train[i] for i in data_index]
                self._label_list_train = [self._label_list_train[i] for i in data_index]
            else:
                self._merge_image_train = [self._merge_image_train[i] for i in data_index]
                self._label_list_train = [self._label_list_train[i] for i in data_index]
            start = 0
            self._index_in_epoch = self._batch_size
            assert self._batch_size <= self._sample_sum_train
        end = self._index_in_epoch
        if self._method != 2:
            image = self._image_list_train[start:end]
        else:
            image = self._merge_image_train[start:end]
        label = np.asarray(self.one_hot(self._label_list_train[start:end]))
        return {'image': image, 'label': label}

    def val_batch(self):
        # 返回所有的val数据
        if self._method != 2:
            return {'image': self._image_list_val, 'label': np.asarray(self.one_hot(self._label_list_val))}
        else:
            return {'image': self._merge_image_val, 'label': np.asarray(self.one_hot(self._label_list_val))}

    def test_batch(self):
        # 返回所有的test数据
        if self._method != 2:
            return {'image': self._image_list_test, 'label': np.asarray(self.one_hot(self._label_list_test))}
        else:
            return {'image': self._merge_image_test, 'label': np.asarray(self.one_hot(self._label_list_test))}

    def train_batch(self):
        # 返回所有的test数据
        if self._method != 2:
            return {'image': self._image_list_train, 'label': np.asarray(self.one_hot(self._label_list_train))}
        else:
            return {'image': self._merge_image_test, 'label': np.asarray(self.one_hot(self._label_list_test))}



if __name__ == '__main__':
    filepath = '../../data/4-training_data/cnn_train/'
    feature_list = ['petrel_Time_gain_attr',
                    'petrel_Dip_deviation_attr',
                    'petrel_Amplitude_contrast_attr',
                    'petrel_chaos_attr']
    input_size = 5
    image_reader = ImageReader(filepath,feature_list, input_size, batch_size=32,partition=[0.6,0.2,0.2],method=2,norm='min_max')
    print(image_reader.sample_num_train,image_reader.sample_num_val,image_reader.sample_num_test)
    for i in range(5):
        labels = image_reader.next_batch()['label']
        images = image_reader.next_batch()['image'][0]
        print(images)
        print(images.shape)
        exit()