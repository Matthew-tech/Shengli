#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-26 下午9:25
# @Author  : zejin
# @File    : BiRNN_dynamic.py
from models.DLmodel.model.point_to_label.Config import model_config
import tensorflow as tf


def variable_on_cpu(name, shape, initializer):
    """
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def BiRNN(X,seq_length,max_len, name=None):
    with tf.variable_scope(name) as scope:
        #if Cur_model_count>0:
        #    scope.reuse_variables()
        with tf.name_scope('lstm'):
            # Forward direction cell:
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(model_config.CELLSIZE, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                         input_keep_prob=1.0 - 0.4,
                                                         output_keep_prob=1.0 - 0.4,
                                                         # seed=random_seed,
                                                         )
            #lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*model_config.NLAYERS)

            # Backward direction cell:
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(model_config.CELLSIZE, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                         input_keep_prob=1.0 - 0.4,
                                                         output_keep_prob=1.0 - 0.4,
                                                         # seed=random_seed,
                                                         )
            #lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*model_config.NLAYERS)
            # Now we feed X into the LSTM BRNN cell and obtain the LSTM BRNN output.
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                     cell_bw=lstm_bw_cell,
                                                                     inputs=X,
                                                                     dtype=tf.float32,
                                                                     #time_major=True,
                                                                     sequence_length=seq_length)

            tf.summary.histogram("activations", outputs)

            # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
            # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
            outputs = tf.concat(outputs, 2)
            outputs = tf.reshape(outputs, [-1, 2 * model_config.CELLSIZE])   # [n_steps*batch_size, 2*n_cell_dim]
        with tf.name_scope('fc5'):
            # Now we feed `outputs` to the  hidden layer with clipped RELU activation and dropout
            b5 = variable_on_cpu('b5', [model_config.OUTPUT_DIM], tf.random_normal_initializer(stddev=0.01))
            h5 = variable_on_cpu('h5', [(2 * model_config.CELLSIZE), model_config.OUTPUT_DIM],tf.random_normal_initializer(stddev=0.01))
            # layer_fc1 = selu(tf.add(tf.matmul(outputs, h5), b5))
            layer_fc1 = tf.nn.softmax(tf.add(tf.matmul(outputs, h5), b5))
            # layer_fc1 = tf.nn.dropout(layer_fc1, (1.0 - 0.3))

            tf.summary.histogram("weights", h5)
            tf.summary.histogram("biases", b5)
            tf.summary.histogram("activations", layer_fc1)
        layer_fc1 = tf.reshape(layer_fc1, [-1, max_len, model_config.OUTPUT_DIM])
        summary_op = tf.summary.merge_all()
        return layer_fc1,seq_length, summary_op
def multi_layer_birnn(inputs, seq_lengths,max_len,layers=1,keep_prob = 0.3,cellsize = 32,rnn_cell = 'LSTM',name='multi_layer_birnn'):
    """
    multi layer bidirectional rnn
    :param RNN: RNN class, e.g. LSTMCell
    :param num_units: int, hidden unit of RNN cell
    :param num_layers: int, the number of layers
    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
    :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size
    :param batch_size: int
    :return: the output of last layer bidirectional rnn with concatenating
    这里用到几个tf的特性
    1. tf.variable_scope(None, default_name="bidirectional-rnn")使用default_name
    的话,tf会自动处理命名冲突
    """
    # TODO: add time_major parameter, and using batch_size = tf.shape(inputs)[0], and more assert
    with tf.variable_scope(name) as scope:
        _inputs = inputs
        if len(_inputs.get_shape().as_list()) != 3:
            raise ValueError("the inputs must be 3-dimentional Tensor")
        for _ in range(layers):
            # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
            # 恶心的很.如果不在这加的话,会报错的.
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                if rnn_cell == 'LSTM':
                    rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(cellsize, forget_bias=1.0, state_is_tuple=True)
                elif rnn_cell == 'GRU':
                    rnn_cell_fw = tf.contrib.rnn.GRUCell(cellsize)
                rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(rnn_cell_fw,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob,
                                                             )
                if rnn_cell == 'LSTM':
                    rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(cellsize, forget_bias=1.0, state_is_tuple=True)
                elif rnn_cell == 'GRU':
                    rnn_cell_bw = tf.contrib.rnn.GRUCell(cellsize)
                rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(rnn_cell_bw,
                                                             input_keep_prob=keep_prob,
                                                             output_keep_prob=keep_prob,
                                                             )
                #initial_state_fw = rnn_cell_fw.zero_state(model_config.BATCHSIZE, dtype=tf.float32)
                #initial_state_bw = rnn_cell_bw.zero_state(model_config.BATCHSIZE, dtype=tf.float32)
                #(output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                #                                              initial_state_fw, initial_state_bw, dtype=tf.float32)
                (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,dtype=tf.float32)

            _inputs = tf.concat(output, 2)
        outputs = tf.reshape(_inputs, [-1, 2 * cellsize])  # [n_steps*batch_size, 2*n_cell_dim]
        with tf.name_scope('fc5'):
            # Now we feed `outputs` to the  hidden layer with clipped RELU activation and dropout
            b5 = variable_on_cpu('b5', [model_config.OUTPUT_DIM], tf.random_normal_initializer(stddev=0.01))
            h5 = variable_on_cpu('h5', [(2 * cellsize), model_config.OUTPUT_DIM],
                                 tf.random_normal_initializer(stddev=0.01))
            # layer_fc1 = selu(tf.add(tf.matmul(outputs, h5), b5))
            layer_fc1 = tf.nn.softmax(tf.add(tf.matmul(outputs, h5), b5))

            tf.summary.histogram("weights", h5)
            tf.summary.histogram("biases", b5)
            tf.summary.histogram("activations", layer_fc1)
        layer_fc1 = tf.reshape(layer_fc1, [-1, max_len, model_config.OUTPUT_DIM])
        return layer_fc1,seq_lengths
if __name__ == '__main__':
    X = tf.placeholder(shape=[None, 240, 76], dtype=tf.float32,
                       name='input_x')
    #X = tf.transpose(X, [1, 0, 2], name='input_x')  # [max_time, batch_size, input_dim]
    seq_length = tf.placeholder(shape=[None], dtype=tf.int32, name='seq_length')
#    output, seq_length, summary_op = BiRNN(X=X, seq_length=seq_length, max_len=240,name='BiRNN_' + str(1))
    keep_prob = tf.placeholder(tf.float32)
    output = multi_layer_birnn(inputs=X,seq_lengths=seq_length,max_len=240,layers=1,keep_prob=keep_prob,
                               cellsize=16,rnn_cell='GRU',name='multi_layer_birnn')
    print(output)