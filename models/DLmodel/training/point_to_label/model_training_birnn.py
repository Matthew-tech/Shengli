#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-26 下午9:18
# @Author  : Eric
# @File    : model_training_birnn_dynamic.py

import csv
import shutil
import tensorflow as tf
from models.DLmodel.model.point_to_label.Config import files_deep, model_config  # paras_record
from models.DLmodel.model.point_to_label.BiRNN_dynamic import *
from statistic.evaluate import evaluate
import matplotlib.pyplot as plt
from data_prepare.point_to_label.get_input_data_p2l import *
import multiprocessing
import pickle as pkl
import pandas as pd
import os
from Configure.global_config import file_loc_gl

def plot_relation_error_f1(source_filename=None,use_alllabels=False,use_split=False):
    file_dir = ''
    with open(os.path.join(file_dir,source_filename),'r') as file:
        data = pd.read_csv(file)
        # 将data按照 data['error'] 进行排序
        data_use_alllabels_index = data['use_alllabel'].values
        #print(data_use_alllabels_index)
        if use_alllabels:
            data_part_index = [i for i,j in enumerate(data_use_alllabels_index) if j]
        else:
            data_part_index = [i for i, j in enumerate(data_use_alllabels_index) if not j]
        # 取出其中use_all_labels = True 的参数组合
        data_part = data.iloc[data_part_index]
        data_part = data_part.sort_values(axis=0, ascending=True,by='error')
        error_list = data_part['error'].values
        row_index = data_part['error'].index
        f1_15 = data_part['1/5'].values
        f1_14 = data_part['1/4'].values
        f1_13 = data_part['1/3'].values
        f1_12 = data_part['1/2'].values
        f1_11 = data_part['1'].values
        #paras = ['%g_layer' % (data_part['layers'][i]) for i in row_index]
        paras = ['%g layer %s      %g %g %s'%(data_part['layers'][i],data_part['normalize'][i],data_part['cellsize'][i],
                 data_part['dropout'][i],data_part['rnn_cell'][i]) for i in row_index]
        stat_list = [f1_15, f1_14, f1_13, f1_12, f1_11]
        fontsize = 20
        fig = plt.figure(figsize=(25,18))
        ax1 = fig.add_subplot(111)
        ax1.plot(error_list,'k',label='Cross Entropy Loss')
        ax1.legend(loc=(.02,.94),fontsize=16)
        #plt.legend(['Cross Entropy Loss'])
        if use_alllabels:
            title_added = '(use all labels,split:%s)'%str(use_split)
        else:
            title_added = '(use target segment labels,split:%s)'%str(use_split)
        ax1.set_title('Cross Entropy Loss and F1_measure '+title_added,fontsize=fontsize)
        ax1.set_ylabel('Cross Entropy Loss',fontsize=fontsize)
        if not use_split:
            ax1.set_ylim([0,1])
        ax2 = ax1.twinx()
        ax2_labels = ['F1_15','F1_14','F1_13','F1_12','F1_11']
        for i,list_ in enumerate(stat_list):
            ax2.plot(list_,label = ax2_labels[i])
        ax2.set_ylabel('F1_measure',fontsize=fontsize)
        ax2.set_xlabel('Paras No.',fontsize=fontsize)
        if not use_split:
            ax2.set_ylim([0,1])
        ax2.legend(loc=(.02,.7),fontsize=16)
        #plt.legend()
        for i,error in enumerate(error_list):
            if not use_split:
                y_loc = error+0.11
            else:
                y_loc = error+0.22
            ax1.text(i,y_loc,paras[i],rotation=90)
        pic_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/pics')
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(os.path.join(pic_dir,'error_f_curve.png'),dpi=100)
        #plt.show()
def predict_training(train_x, train_y, validation_x, validation_y, ts_max, paras, test=False, save=False):
    """
    :param train_x:
    :param train_y:
    :param validation_x:
    :param validation_y:
    :param ts_max: 目标层段的最大长度
    :param paras:
    :paras = {'model': model_type, 'trace_range': trace_range, 'normalize': normalize,
              'opt': 'Adam', 'loss': 'CEE', 'layers': layers, 'cellsize':cellsize,
              'dropout':dropout,'rnn_cell':rnn_cell,'count': count}
    :param test: True or False 表示进行模型选择还是进行测试
    :param save: 表示是否保存权重文件
    :return:
    """
    cellsize = paras['cellsize']
    rnn_cell = paras['rnn_cell']

    train_batches = BatchGenerator_p2l(train_x, train_y, batch_size=model_config.BATCHSIZE,
                                       ts_max=ts_max)  # {'feature','label','seq_len'}
    validation_batches = BatchGenerator_p2l(validation_x, validation_y, batch_size=len(validation_x), ts_max=ts_max)
    batch_ex = train_batches.next_batch()

    # [batchsize,max_time,input_dim]
    X_len = tf.placeholder(dtype=tf.int32, name='X_len')
    X = tf.placeholder(shape=[None, None, train_batches.feature_dim], dtype=tf.float32,
                       name='input_x')
    seq_length = tf.placeholder(shape=[None], dtype=tf.int32, name='seq_length')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    max_len = train_batches.seq_len_max
    layers = paras['layers']
    if paras['model'] == 'birnn':
        name = 'BiRNN_' + str(paras['count'])
        X = tf.transpose(X, [1, 0, 2], name='input_x')  # [max_time, batch_size, input_dim]
        output, seq_length, summary_op = BiRNN(X=X, seq_length=seq_length, max_len=max_len, name=name)
    elif paras['model'] == 'multi_layer_birnn':

        name = 'multi_layer_BiRNN_' + str(paras['count'])
        output, seq_length = multi_layer_birnn(inputs=X, seq_lengths=seq_length, max_len=X_len, keep_prob=keep_prob,
                                               layers=layers, cellsize=cellsize, rnn_cell=rnn_cell, name=name)
    # 真实标签
    y_data = tf.placeholder(shape=[None, None, model_config.OUTPUT_DIM], dtype=tf.float32)
    if paras['loss'] == 'MSE':
        loss = tf.reduce_mean(tf.square(output - y_data))
    elif paras['loss'] == 'CEE':
        loss = -tf.reduce_mean(y_data * tf.log(output))
    if paras['trace_range'] == 0:
        nb_epoch = 1000
    elif paras['trace_range'] == 1:
        nb_epoch = 200
    elif paras['trace_range'] == 2:
        nb_epoch = 70
    iterations = int((train_batches.samples_num / model_config.BATCHSIZE) * nb_epoch)
    current_iter = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.0, current_iter, decay_steps=iterations, decay_rate=0.03)
    if paras['opt'] == 'GD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif paras['opt'] == 'Adag':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif paras['opt'] == 'Mom':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif paras['opt'] == 'RMS':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif paras['opt'] == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # train = optimizer.minimize(loss, global_step=current_iter)
    train = optimizer.minimize(loss)
    # 创建 saver，用来保存模型和参数
    saver = tf.train.Saver()
    tf.add_to_collection('predict_network', output)
    with tf.Session() as sess:
        # BiRNN

        sess.run(tf.global_variables_initializer())
        train_loss_list = []
        validation_loss_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        # iterations = 1
        for step in range(iterations):
            Cur_data = train_batches.next_batch()
            X_input, y_data_input = Cur_data['feature'], Cur_data['label']
            Cur_maxlen = max(Cur_data['seq_len'])
            X_input = X_input[:, :Cur_maxlen, :]
            y_data_input = y_data_input[:, :Cur_maxlen, :]

            if paras['model'] == 'birnn':
                X_input = X_input.transpose([1, 0, 2])

            feed_dict = {X: X_input, X_len: Cur_maxlen, y_data: y_data_input, seq_length: Cur_data['seq_len'],
                         keep_prob: 1 - paras['dropout']}
            Cur_loss, _, y_output, lr = sess.run(fetches=[loss, train, output, learning_rate], feed_dict=feed_dict)
            train_loss_list.append(Cur_loss)

            if paras['model'] == 'birnn':
                valid_x_input = valid_x_input.transpose([1, 0, 2])
            # 在validation上面验证一下
            validation_data = validation_batches.next_batch()
            valid_x_input, valid_y_input = validation_data['feature'], validation_data['label']
            Cur_maxlen = max(validation_data['seq_len'])
            X_input = valid_x_input[:, :Cur_maxlen, :]
            y_data_input = valid_y_input[:, :Cur_maxlen, :]
            feed_dict = {X: X_input, X_len: Cur_maxlen, y_data: y_data_input,
                         seq_length: validation_data['seq_len'], keep_prob: 1}
            Cur_loss_val, y_output_val = sess.run(fetches=[loss, output], feed_dict=feed_dict)
            validation_loss_list.append(Cur_loss_val)
            acc, recall, f1, sum_ = evaluate(valid_y_input, y_output_val, validation_data['seq_len'], test=test)
            f1_list.append(f1)
            precision_list.append(acc)
            recall_list.append(recall)
            if step % 30 == 0:
                print(step, '/', iterations,
                      'Cur_loss_val:%g, acc:%g, recall:%g, f1:%g ' % (Cur_loss_val, acc, recall, f1), sum_)
        if test:
            # 保存 train_loss 和 validation_loss
            plt.plot(train_loss_list)
            plt.plot(validation_loss_list)
            fontsize = 16
            plt.legend(['Training Loss','Validation Loss'],fontsize=fontsize)
            plt.xlabel('Iteration',fontsize=fontsize)
            plt.ylabel('Cross Entropy Loss',fontsize=fontsize)
            plt.title('Training Loss and Validation Loss')
            fig_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/pics')
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(os.path.join(fig_dir,'train_val_loss.png'),dpi=200)

            # 在test上面验证一下,并分别计算其PR曲线和ROC曲线(这里的validation 就是test)
            test_data = validation_batches.next_batch()
            test_x_input, test_y_input = test_data['feature'], test_data['label']
            Cur_maxlen = max(test_data['seq_len'])
            X_input = test_x_input[:, :Cur_maxlen, :]
            y_data_input = test_y_input[:, :Cur_maxlen, :]

            feed_dict = {X: X_input, X_len: Cur_maxlen, y_data: y_data_input, seq_length: test_data['seq_len'],
                         keep_prob: 1}
            Cur_loss_test, y_output_test = sess.run(fetches=[loss, output], feed_dict=feed_dict)
            Precision, Recall, F1, FPR = evaluate(test_y_input, y_output_test, test_data['seq_len'], test=test)
            fig = plt.figure(figsize=(25,18))
            ax1 = fig.add_subplot(211)
            ax1.plot(Recall, Precision)
            fontsize = 16
            ax1.set_xlabel('Recall',fontsize=fontsize)
            ax1.set_ylabel('Precision',fontsize=fontsize)
            ax1.set_title('PR Curve',fontsize=fontsize)

            ax2 = fig.add_subplot(212)
            ax2.plot(FPR,Recall)
            ax2.plot([0,1],[0,1],'r--')
            ax2.set_xlabel('False Positive Rate',fontsize=fontsize)
            ax2.set_ylabel('True Positive Rate',fontsize=fontsize)
            ax2.set_title('ROC Curve',fontsize=fontsize)
            
            plt.savefig(os.path.join(fig_dir,'roc_pr.png'),dpi=200)
            
        if save:
            print('正在保存权重文件...')
            weight_dir = file_loc_gl.weights_BiRNN
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)
            weight_name = paras_weight_name(paras)
            weight_path = os.path.join(weight_dir, weight_name)
            saver.save(sess, weight_path)  # 保存图
        if not test:
            result = [np.mean(validation_loss_list[-30:]), np.mean(precision_list[-30:]), np.mean(recall_list[-30:])]
            save_result(paras, result)
        else:
            return [Precision, Recall, F1, FPR], [test_data['seq_len'], test_y_input, y_output_test]


def save_result(paras, result):
    results = []
    results.append(
        [paras["count"], paras["model"], paras["layers"], paras["trace_range"], paras["normalize"], paras["cellsize"],
         paras["dropout"], paras["rnn_cell"], paras["use_alllabel"], result[0], result[1], result[2]])

    with open(os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/model_selection_temp_results/') +
                      params2name(paras) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(['count', 'model', 'layers', 'trace_range', 'normalize', 'cellsize', 'dropout', 'rnn_cell',
                         'use_alllabel', 'error', 'precision', 'recall'])
        for row in results:
            writer.writerow(row)


def model_selection_training():
    # 创建临时保存BiRNN结果的目录
    model_selection_result = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/model_selection_temp_results')
    if os.path.isdir(model_selection_result):
        shutil.rmtree(model_selection_result)
        os.mkdir(model_selection_result)
    else:
        os.mkdir(model_selection_result)

    all_labels = get_labels()
    count = 1
    model_type = 'multi_layer_birnn'
    pool = multiprocessing.Pool()
    for trace_range in [0]:
        for normalize in ['GN', 'MN']:
            all_seismic_data = get_seismic_data(trace_range=trace_range, normalize=normalize)
            train_o, validation_o, test_o, ts_max_o = split_data(all_seismic_data, all_labels, train=0.6,
                                                                 validation=0.2, test=0.2)
            train, validation, test, ts_max = train_o.copy(), validation_o.copy(), test_o.copy(), ts_max_o

            for layers in [1, 2, 3]:
                for cellsize in [16, 32, 64]:
                    for dropout in [0.3, 0.4, 0.5]:
                        for rnn_cell in ['GRU', 'LSTM']:
                            for use_alllabel in [True, False]:  # 两个的参数对应的标记不同，不能简单比较
                                paras = {'model': model_type, 'trace_range': trace_range, 'normalize': normalize,
                                         'opt': 'Adam', 'loss': 'CEE', 'layers': layers, 'cellsize': cellsize,
                                         'dropout': dropout, 'rnn_cell': rnn_cell, 'use_alllabel': use_alllabel,
                                         'count': count}
                                print("Count: ", count)
                                count += 1
                                pool.apply_async(predict_training, args=(
                                train[0], train[1], validation[0], validation[1], ts_max, paras,))
    pool.close()
    pool.join()


def paras_weight_name(paras):
    """
    根据参数组合生成相应的权重文件的名字
    """
    weight_name = 'weights_tracerange_%g_layer_%g_norm_%s_cell_%g_dropout_%g_%s_ts_%s' %(paras['trace_range'], paras['layers'], paras['normalize'], paras['cellsize'],
     paras['dropout'], paras['rnn_cell'], str(paras['use_alllabel']))
    return weight_name


def params2name(paras):
    name = ""
    for k in paras:
        name += k + "_" + str(paras[k]) + "_"
    return name


def model_evaluation(save_graph=True):
    """
    对模型进行评价，
    :param save:  如果save = True，表示保存权重文件
    :return:      None
    """
    print('开始模型选择')
    # 确定模型最优参数组合
    model_selection_training()
    merge_results()
    best_paras = paras_selection()
    paras = {}
    paras['count'] = int(best_paras[0])
    paras['model'] = best_paras[1]
    paras['layers'] = int(best_paras[2])
    paras['trace_range'] = int(best_paras[3])
    paras['normalize'] = best_paras[4]
    paras['cellsize'] = int(best_paras[5])
    paras['dropout'] = float(best_paras[6])
    paras['rnn_cell'] = best_paras[7]
    paras['use_alllabel'] = best_paras[8]
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

    if save_graph:
        train_p = 0.7;
        validation_p = 0.2;
        test_p = 0.1
    else:
        train_p = 0.7;
        validation_p = 0.0;
        test_p = 0.3

    all_labels = get_labels(paras['use_alllabel'])  # use_alllabel = True 表示使用全部的标记数据，False表示只使用目标层段
    all_seismic_data = get_seismic_data(trace_range=paras['trace_range'], normalize=paras['normalize'])
    print(paras)
    train, validation, test, ts_max = split_data(all_seismic_data, all_labels, train=train_p,
                                                 validation=validation_p, test=test_p)

    # test[0] 是features，test[1] 是labels，test[2] 是深度信息
    results, labels = predict_training(train[0], train[1], test[0], test[1], ts_max, paras, test=True, save=save_graph)
    print('保存图')
    if not save_graph:
        # 保存统计结果
        results_dir = os.path.join(file_loc_gl.results_root,'test_wells_result/')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        evaluate_name = 'evaluation_' + str(paras['count']) + '.csv'
        with open(os.path.join(results_dir, evaluate_name), 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel')
            writer.writerow(['threshold', 'precision', 'recall', 'f1', 'fpr'])
            for i in range(0, 101):
                threshold = i * 0.01
                writer.writerow([threshold, results[0][i], results[1][i], results[2][i], results[3][i]])

        # 保存预测结果，并保存每个点的深度信息
        result_name = 'results_' + str(paras['count']) + '_add_depth.csv'
        with open(os.path.join(results_dir, result_name), 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel')
            # writer.writerow(['samples num',len(labels[0]),'points num',sum(labels[0])])
            writer.writerow(['No', 'depth', 'true_label', 'predict_label'])
            for i, Cur_len in enumerate(labels[0]):
                No_start = 1
                depth_start = test[2][i][0]
                for tlabel, plabel in zip(labels[1][i][:Cur_len], labels[2][i][:Cur_len]):
                    writer.writerow([No_start, depth_start, tlabel[1], plabel[1]])
                    No_start += 1
                    depth_start += 1
    print('模型选择完成')


def paras_selection():
    """
    根据临时结果选择最优参数组合
    """
    file_path = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/model_selection_temp_results/merge_results.csv')
    plot_relation_error_f1(file_path)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            data = csv.reader(f)
            # 基于 error 按升序排序
            sortedList = sorted(data, key=lambda x: (x[9]))
            best_paras = sortedList[0]
            print(best_paras)
            return best_paras


def merge_results():
    """
    将不同参数组对应的临时结果进行汇总，确定最优参数组合
    """
    file_dir = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/model_selection_temp_results')
    files = os.listdir(file_dir)
    lines = []
    i = 0
    for file in files:
        file_path = os.path.join(file_dir, file)
        if ".csv" in file:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for n, l in enumerate(reader):
                    if i < 1:
                        lines.append(l)
                    elif n > 0:
                        lines.append(l)
                i += 1

    beta_str = ['1/5', '1/4', '1/3', '1/2', '1', '2', '3', '4', '5']
    beta = [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5]

    for n, line in enumerate(lines):
        if n == 0:
            # 标题
            for b in beta_str:
                line.append(b)
        elif n > 0:
            p = float(line[-2])
            r = float(line[-1])
            for b in beta:
                f = (p * r * (1 + b * b)) / (b * b * p + r)
                line.append(f)

    # 清空保存临时结果目录下的文件
    model_selection_temp_result = os.path.join(file_loc_gl.results_root,'point_to_label/BiRNN/model_selection_temp_results')
    if os.path.isdir(model_selection_temp_result):
        shutil.rmtree(model_selection_temp_result)
        os.mkdir(model_selection_temp_result)
    else:
        os.mkdir(model_selection_temp_result)
    with open(os.path.join(model_selection_temp_result, 'merge_results.csv'), 'w',
              newline='') as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)


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
    model_evaluation()
    #best_paras_pkl()
    print("model training finish!")
