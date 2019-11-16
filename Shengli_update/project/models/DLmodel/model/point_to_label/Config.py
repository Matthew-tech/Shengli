from Configure.global_config import *
import os


reservoir_ranges = [0,1,2]
feature_op_types = ['average']  # ['add', 'average', 'weighted_add', 'weighted_average', 'merge']
is_use_target_segments = [True]
test_part = 0.3
nb_epoch = 5
feature_num = 11
eval_cost_type = 'mse'
feature_type = 'origin'         # 'spectrum'
model_num = 500

class data_process_s2l():
    def __init__(self):
        self.big_window_ol = 2                   # 大时窗重叠长度
        self.small_window_ol = 10                # 小时窗重叠长度
        #data_aug = [big_window,big_window_ol,small_window_ol]   # 300 是大时窗， 2是大时窗重叠的长度，10 是小时窗重叠的长度
        self.seg_Intervals = [30]                                    # 小时窗的大小，有 30 50
        self.is_normalize = False                 # 是否将1251个点进行归一化（min_max 归一化方法）
        self.trace_range = 1

data_config_s2l = data_process_s2l()

class model_paras():
    def __init__(self):
        if data_config_s2l.seg_Intervals[0] == 30:
            self.SEQ_LEN = 15  # 15(30ms), 8 (50ms)
            self.INPUT_DIM = 15
            self.big_window = 310
        elif data_config_s2l.seg_Intervals[0] == 50:
            self.SEQ_LEN = 8
            self.INPUT_DIM = 25
            self.big_window = 330
        self.model_type =  'HMM'#'BidLSTM'# 'BiRNN'#          # 'multi_layers_lstm', 'BiRNN', 'seq2seq'
        self.OUTPUT_DIM = 2
        self.DATA_INPUT_DIM = self.INPUT_DIM
        self.BATCHSIZE = 32
        # tensorflow hyperparameters
        self.CELLSIZE = 32
        self.NLAYERS = 2
        self.BiRNN_LAYERS = 1
        self.activation_func = 'SELU'  # sigmoid, SELU, RELU
        self.selu_dropout_prob = 0.3
        self.model_graph_weights = 'models/DLmodel/model_graph_weights'
model_config = model_paras()


train_type = 'seg_to_label'#'point_to_label'       # 使用点到label的预测  还包括 'seg_to_label' 方式
training_param_combination = 'train_type_'+train_type+'_seg_' + str(data_config_s2l.seg_Intervals[0]) + '_' + \
                             model_config.model_type +'_trace_range_' \
                             + str(data_config_s2l.trace_range) + '_shuffle_wells_' + str(SHUFFLE_ALL_WELLS)
# file location
class files_loc_deep():     # 深度学习用文件位置
    def __init__(self):

        self.training_data_dir = 'data/4-training_data/'
        #self.saveFilePath_Base = 'data/test_full_data/'

        # 预测使用的文件或文件夹
        # self.seismic_data_dir = 'data/3-seismic_data'
        self.seismic_data_dir = file_loc_gl.seismic_sgy_file_path_base#'/disk2/Shengli/data/seismicdata'#修改
        self.full_train_data = 'data/full_train_data'
        # 相关性
        self.high_correlation_saved_file = os.path.join(self.full_train_data,'high_correlation_attrs.pkl')
        self.train_data_dir = os.path.join(self.training_data_dir,'normalize_'+str(data_config_s2l.is_normalize))
        self.test_data_dir = os.path.join('data/5-testing_data','normalize_'+str(data_config_s2l.is_normalize))
        self.low_correlation_dir = 'data/low_correlation'

        # self.weights_dir = os.path.join(model_config.model_graph_weights,training_param_combination)
        self.weights_dir = model_config.model_graph_weights

        # 位置标记文件
        self.shuffle_all_wells = 'models/DLmodel/intermediate_index/shuffle_all_wells_index.pkl'
        self.shuffle_all_big_windows = 'models/DLmodel/intermediate_index/shuffle_all_big_windows_index_'+\
                                       training_param_combination+'.pkl'
        self.test_big_window_index = 'models/DLmodel/intermediate_index/test_big_windows_index.pkl'

files_deep = files_loc_deep()


