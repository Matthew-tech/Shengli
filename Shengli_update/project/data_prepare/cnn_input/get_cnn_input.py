import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
class ImageReader(object):
    def __init__(self,data_dir,batch_size,partition,channel=3,norm='z_score'):
        print('initializing...')
        self._trace_step = 14
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._partition = partition
        self._channel = channel
        self._norm = norm
        self._train = self._partition[0];self._val = self._partition[1];self._test = self._partition[2]
        for filename in os.listdir(self._data_dir):
            if 'pos' in filename:
                self._pos_filename = os.path.join(self._data_dir,filename)
            elif 'neg' in filename:
                self._neg_filename = os.path.join(self._data_dir,filename)
        with open(self._pos_filename,'rb') as file1,open(self._neg_filename,'rb') as file2:
            self._data_pos = pickle.load(file1)
            self._data_neg = pickle.load(file2)
        self._wells = list(set(list(self._data_pos.keys())+list(self._data_neg.keys())))
        random_index = list(range(len(self._wells)))
        random.shuffle(random_index)
        self._wells_train = self._wells[:int(len(self._wells)*self._train)]
        self._wells_val = self._wells[int(len(self._wells) * self._train):int(len(self._wells) * (self._train+self._val))]
        self._wells_test = self._wells[int(len(self._wells) * (self._train+self._val)):]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._image_list_train,self._label_list_train=self.get_image_data(type_='train',norm=self._norm)
        self._image_list_val, self._label_list_val = self.get_image_data(type_='val',norm=self._norm)
        self._image_list_test, self._label_list_test = self.get_image_data(type_='test',norm=self._norm)
        self._sample_sum_train = len(self._image_list_train)
        self._sample_sum_val = len(self._image_list_val)
        self._sample_sum_test = len(self._image_list_test)
    def normalize(self,data,norm='z_score'):
        """
        对数据进行归一化，data 是要归一化的数据，shape = (28,28,3), norm 是归一化的方式
        :param data:
        :param norm:
        :return:
        """
        norm_paras_file = os.path.join('../../', 'data/full_train_data/max_min_mean_std_new.pkl')
        with open(norm_paras_file, 'rb') as file:
            paras = pickle.load(file)  # {filename:[max, min, mean, std] }
            for key in paras.keys():
                if 'CDD' in key:
                    Cur_paras = paras.get(key)
                    break
        if norm == 'z_score':
            image = (data - Cur_paras[2])/Cur_paras[3]
        elif norm == 'min_max':
            image = (data - Cur_paras[1])/(Cur_paras[0]-Cur_paras[1])
        return image

    def turn_into_specific_channel(self,image,norm='z_score'):
        image = self.normalize(image,norm=norm)
        if self._channel == 3:
            return image
        elif self._channel == 1:
            temp = np.zeros(shape=(self._trace_step*2, self._trace_step*2))
            temp[:self._trace_step, :self._trace_step] = image[:, :, 0]
            temp[:self._trace_step, self._trace_step:] = image[:, :, 1]
            temp[self._trace_step:, :self._trace_step] = image[:, :, 2]
            return temp.reshape((self._trace_step*2,self._trace_step*2,1))
    def get_image_data(self,type_='train',norm='z_score'):
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
                    images_list.append(self.turn_into_specific_channel(list(image_info.values())[0],norm=norm))
                    labels_list.append(1)
        # 将负样本放进去
        for key in self._data_neg.keys():
            if key in Cur_wells:
                for image_info in self._data_neg.get(key):
                    images_list.append(self.turn_into_specific_channel(list(image_info.values())[0],norm=norm))
                    labels_list.append(0)

        random_index = list(range(len(images_list)))
        random.shuffle(random_index)
        images_list = [images_list[i] for i in random_index]
        labels_list = [labels_list[i] for i in random_index]
        return images_list, labels_list
    @property
    def wells_train(self):  return self._train
    @property
    def wells_val(self):    return self._val
    @property
    def wells_test(self):   return self._val
    @property
    def chanel(self):   return self._channel
    @property
    def sample_num_train(self): return self._sample_sum_train
    @property
    def sample_num_val(self):   return self._sample_sum_val
    @property
    def sample_num_test(self):  return self._sample_sum_test
    def get_all_data(self,shuffle=True):
        # 得到所有的训练数据
        pass
    def one_hot(self,labels):
        print(labels)
        for i,sample in enumerate(labels):
            labels[i] = [0,1] if sample == 1 else [1,0]
        print('after:',labels)
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
            self._image_list_train = [self._image_list_train[i] for i in data_index]
            self._label_list_train = [self._label_list_train[i] for i in data_index]
            start = 0
            self._index_in_epoch = self._batch_size
            assert self._batch_size <= self._sample_sum_train
        end = self._index_in_epoch
        image = self._image_list_train[start:end]
        label = np.asarray(self.one_hot(self._label_list_train[start:end]))
        return {'image': image, 'label': label}
    def val_batch(self):
        # 返回所有的val数据
        return {'image':self._image_list_val,'label':np.asarray(self.one_hot(self._label_list_val))}
    def test_batch(self):
        # 返回所有的test数据
        return {'image':self._image_list_test,'label':np.asarray(self.one_hot(self._label_list_test))}
if __name__ == '__main__':
    filepath = '../../data/4-training_data/cnn_train/'

    image_reader = ImageReader(filepath,batch_size=32,partition=[0.6,0.2,0.2],channel=1,norm='z_score')
    print(image_reader.sample_num_train,image_reader.sample_num_val,image_reader.sample_num_test)
    for i in range(5):
        labels = image_reader.next_batch()['label']
        images = image_reader.next_batch()['image']
        print(images[0].shape)
        plt.imshow(np.transpose(images[0],axes=[2,0,1]))
        plt.show()
        exit()

