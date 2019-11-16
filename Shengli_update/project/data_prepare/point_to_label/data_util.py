"""
AUTHER: Eric
Date: 2017.07.28
DESCRIPTION:
    # 保存从地震题中抽取的所有数据的特征
    # data_aug 里面的参数表示做 data_augmentation 的时候窗口大小(ms)滑动步长(ms),第三个元素表示每个滑动窗口内各小滑动窗口的重叠长度
    # 生成的文件格式：
    # train_feature_label = [well_name,每个井数据扩充之后的数据] × 井数量
    # [每个井数据扩充之后的数据] = [每个井range范围内每个trace扩充之后的数据] × grid point     (grid的排列方式是从左下向右向上)
    # [每个井range范围内每个trace扩充之后的数据] = [每个300ms对应的储层标记信息] × 300ms 大时窗的个数
    # [每个300ms对应的储层标记信息] = [zip(储层段特征list,对应的标记list),[深度顶，深度底]]
"""
import os
import struct
import csv
import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
from decimal import getcontext, Decimal
from Configure.global_config import file_loc_gl
from models.DLmodel.model.point_to_label.Config import files_deep
import pickle
from data_prepare.select_high_correlation_attrs import read_high_correlation_files
sampling_rate = 160


def Build_Interpolation_function(depth_time_rel_filepath, is_reverse = False):
    # depth_time_rel_filepath = 'H:\\研究生\\2-胜利油田\\数据\\1-well\\depth_time_rel\\cb6_ck.txt'
    if not os.path.exists(depth_time_rel_filepath):
        #print('没有找到时深文件:%s'%depth_time_rel_filepath)
        return -1,-1,-1
    with open(depth_time_rel_filepath,'r') as rel_file:
        depth_list = []
        time_list = []
        for line_i in range(3):
            Cur_line = rel_file.readline()
        Cur_line = Cur_line.split(' ')
        Cur_line_clean = []
        for ele in Cur_line:
            if ele == '' or ele == '\n':
                continue
            else:
                Cur_line_clean.append(ele)
        if Cur_line_clean[1] == 'ms':
            time_col_index = 1
            depth_col_index = 2
        else:
            time_col_index = 2
            depth_col_index = 1
        Cur_line = rel_file.readline()
        while(Cur_line):
            Cur_line = Cur_line.split(' ')
            Cur_line_clean = []
            for ele in Cur_line:
                if ele == '' or ele == '\n':
                    continue
                else:
                    Cur_line_clean.append(ele)
           # print(Cur_line_clean)
            depth_list.append(float(Cur_line_clean[depth_col_index]))
            time_list.append(float(Cur_line_clean[time_col_index]))
            Cur_line = rel_file.readline()
        kind = ["nearest", "zero", "slinear", "quadratic", "cubic"]
        selection_number = 2
     #   print('正在生成井：',depth_time_rel_filepath,'的',kind[selection_number],'样条函数...')

        #f = interpolate.interp1d(depth_list, time_list, kind=kind[selection_number]) # 三阶样条插值
        if not is_reverse:
            f = interpolate.UnivariateSpline(depth_list, time_list, k=1,s=2)  # 三阶样条插值
        elif is_reverse:
            f = interpolate.UnivariateSpline(time_list, depth_list, k=1,s=2)  # 三阶样条插值

    return depth_list,time_list,f

def get_well_reservoir_dict(well_reservoir_reader_filepath):
    '''
    得到 well_name 和储层的dict
    :param well_reservoir_reader:
    :return:
    '''
    dict = {}
    well_reservoir_reader = csv.reader(open(well_reservoir_reader_filepath, encoding='utf-8'))
    for row in well_reservoir_reader :
        if row[0] == 'well_no':
            continue
        if row[0] not in dict.keys():
            Cur_reservoir_Info = []
            Cur_reservoir_Info.append([row[0],float(row[1]),float(row[2]),int(row[3])])
            dict.setdefault(row[0],Cur_reservoir_Info)
        else:
            new_value = dict.get(row[0])
            new_value.append([row[0],float(row[1]),float(row[2]),int(row[3])])
            dict.update({row[0]:new_value})
    return dict



def turn2time_Info(well_reservoir_data,rel_filepath=''):
    '''
    进行时深转化
    :param well_reservoir_data: 深度信息, 每个list元素有三个值，分别是well_name，depth_bottom,depth_top
    :return: list，其中每个元素也是一个list
    '''
    if rel_filepath == '':
        depth_time_rel_filepath = '../../data/1-well/depth_time_rel/cb321_ck.txt'
        depth_list, time_list, interpolation_f = Build_Interpolation_function(depth_time_rel_filepath)
    else:
        depth_list, time_list, interpolation_f = Build_Interpolation_function(rel_filepath)
        #print('depth:',depth_list)
        #print('time :',time_list)
        #print('inter:',[float(Decimal.from_float(i).quantize(Decimal('0.00000'))) for i in list(interpolation_f(depth_list))])
        #exit()
    for reservoir_Info in well_reservoir_data:
        if reservoir_Info[1] == 0:
            reservoir_Info[1] = 0
        else:
            reservoir_Info[1] =float(interpolation_f(float(reservoir_Info[1])))
        if reservoir_Info[2] == 0:
            reservoir_Info[2] = 0
        else:
            reservoir_Info[2] = float(interpolation_f(float(reservoir_Info[2])))
    # 将时间 信息 > 2500 的储层删除掉
    well_reservoir_data = [Cur_reservoir for Cur_reservoir in well_reservoir_data if float(Cur_reservoir[1])<=2500 and float(Cur_reservoir[2]<=2500)]
    return well_reservoir_data

def get_specific_reservoir_data(well_reservoir_dt, rel_dir, well_name):
    '''
    :param well_reservoir_dt: list 中的每个元素有三个信息，depth_bottm
    :param rel_dir:
    :param well_name:
    :return:
    '''
    # well_reservoir_dt_only = [[row[0],row[1],row[2]] for row in well_reservoir_dt]
    # 根据well_name在rel_dir中找到对应的时深转换文件，对well_reservoir_dt_only进行时深转换

    if os.path.exists(os.path.join(rel_dir,well_name + '_ck.txt')):
        well_reservoir_data_time = turn2time_Info(well_reservoir_dt,os.path.join(rel_dir,well_name + '_ck.txt'))
    elif os.path.exists(os.path.join(rel_dir , well_name.lower() + '_ck.txt')):
        well_reservoir_data_time = turn2time_Info(well_reservoir_dt,os.path.join(rel_dir,well_name.lower() + '_ck.txt'))
    else:
        print('没有找到井：',well_name,'时深转化文件....')
   # print(well_reservoir_data_time)
    return well_reservoir_data_time
# fft 转化到频域
def get_spectum(sampling_rate,seismic_wave_frag):
    # fft_size = len(seismic_wave_frag)
    fft_size = sampling_rate
    if len(seismic_wave_frag) <= fft_size: #int(sampling_rate/2)+1:
        seismic_wave_frag.extend([0] * (fft_size - len(seismic_wave_frag)))
    else:
        seismic_wave_frag = seismic_wave_frag[:fft_size]

    xf = np.fft.rfft(seismic_wave_frag) / fft_size
    # freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    '''
    print('xpf len:',len(xfp))
    plt.subplot(121)
    plt.plot(seismic_wave_frag)
    plt.xlabel('sampling points')
    plt.subplot(122)
    plt.plot(xfp)
    plt.xlabel('Hz')
    plt.show()
    '''
    return xfp

def get_wave_frag(reservoir_data, mode = 'statistic',seg_interval = 50,overlapping = 0):
    # reservoir_data 中每个顶和底 都是时间表示的
    # test_seg_time 表示 test data中每个 segment 的时长为 50 ms
    # overlapping 表示每个小窗口的冲重叠长度
    num = 1
    b_t_list = []
    if mode == 'statistic':
        label_4_train = []
        # 第一个层段的顶和底
        Cur_bottom = float(reservoir_data[0][1])
        Cur_top = float(reservoir_data[0][2])
        b_t_list.append([Cur_bottom,Cur_top])
        label_4_train.append(int(reservoir_data[0][3]))
        for reservoir in reservoir_data[1:]:
            # 表示中间有断层
            if float(reservoir[1]) != Cur_top:
                num += 2
                b_t_list.append([Cur_top,float(reservoir[1])])
                label_4_train.append(0)
            else:
                num +=1
            Cur_bottom = float(reservoir[1])
            Cur_top = float(reservoir[2])
            b_t_list.append([Cur_bottom,Cur_top])
            label_4_train.append(int(reservoir[3]))
    else:
        # 得到reservoir_data第一个储层的底和最后一个储层的顶
        bottom_most = float(reservoir_data[0][1])
        top_most = float(reservoir_data[-1][2])
        # 可以生成几“段”
        num = int((top_most-bottom_most-seg_interval)/(seg_interval-overlapping))+2
        for i in range(num-1):
            b_t_list.append([bottom_most+(seg_interval-overlapping)*i,bottom_most+(seg_interval-overlapping)*i+seg_interval])
        b_t_list.append([bottom_most+(seg_interval-overlapping)*(num-1),top_most])
    if mode == 'statistic':
        return num, b_t_list,label_4_train
    else:
        return num, b_t_list

def get_label_4_test(b_t_list_4_train, label_4_train, b_t_list):
    # b_t_list_4_train, label_4_train 分别记录按照有标记的层段长度划分的储层 和 其对应的 label
    # b_t_list_4_train  和 b_t_list 一个是不等长划分，一个是等长划分
    label_4_test = []
    interval_loc_train = b_t_list_4_train[0]
    for reservoir in b_t_list_4_train[1:]:
        interval_loc_train.append(reservoir[1])

    interval_loc_test = b_t_list[0]
    for reservoir in b_t_list[1:]:
        interval_loc_test.append(reservoir[1])

    mixed_loc = []
    mixed_loc.extend(interval_loc_train)
    mixed_loc.extend(interval_loc_test[1:])
    mixed_loc = sorted(mixed_loc)
    interval_loc_train_no = 0
    interval_loc_test_no = 1
    Cur_0_len = 0
    Cur_1_len = 0
    for mixed_loc_no in range(len(mixed_loc)-2):
        if mixed_loc[mixed_loc_no+1] < interval_loc_test[interval_loc_test_no]:
            if label_4_train[interval_loc_train_no] == 0:
                Cur_0_len += (mixed_loc[mixed_loc_no+1] - mixed_loc[mixed_loc_no])
            else:
                Cur_1_len += (mixed_loc[mixed_loc_no+1] - mixed_loc[mixed_loc_no])
            # 用来标记标签
            interval_loc_train_no += 1
        else:
            if label_4_train[interval_loc_train_no] == 0:
                Cur_0_len += (interval_loc_test[interval_loc_test_no]-mixed_loc[mixed_loc_no])
            else:
                Cur_1_len += (interval_loc_test[interval_loc_test_no] - mixed_loc[mixed_loc_no])
            label_4_test.append([Cur_0_len,Cur_1_len])
            if label_4_train[interval_loc_train_no] == 0:
                Cur_0_len = mixed_loc[mixed_loc_no+1] - interval_loc_test[interval_loc_test_no]
                Cur_1_len = 0
            else:
                Cur_1_len = mixed_loc[mixed_loc_no+1] - interval_loc_test[interval_loc_test_no]
                Cur_0_len = 0
            interval_loc_test_no += 1
            if len(label_4_test) == len(b_t_list):
                break
    return label_4_test
def normalize(data):
    data = np.asarray(data)
    data = (data - np.min(data))/(np.max(data)-np.min(data))
    return data.tolist()

def get_single_well_lstm_input(all_trace_data,reservoir_data,feature_mode='spectrum',processing_train = False,
                               seg_Interval=50, data_aug = None,is_normalize = False):
    '''
    :param all_trace_data:  list, 包含 当前well 对应的所有trace_data
    :param reservoir_data:  里面的元素有 4 项，分别是 well_name , depth_bottom , depth_top , reservoir_Info，其中的储层顶和底都是用时间表示的
    :param data_aug: 表示是否将数据进行扩充
    :return:
    '''
    # if feature_mode == 'spectum' 根据深度信息得到对应trace_data的频谱图
    # elif feature_mode == 'original' 返回原始振幅数据
    # if processing_train = True, 需要按照储层长度返回feature，
    # if processing_train = False,将储层的长度平分
    # test_seg_time 表示测试层段每个被划分成 多少 毫秒
    # sampling_rate = 160
    fft_size = sampling_rate
    # fft_size = int(seg_Interval/2)
    # wave_frag_num 表示被划分成几段 frag, b_t_list 表示当前井对应的 bottom 和 top 对
    if processing_train:
        wave_frag_num, b_t_list ,_ = get_wave_frag(reservoir_data, processing_train = processing_train, test_seg_time = seg_Interval)
    else:
        # 如果是测试 data，首先生成 train data 的b_t_list 用于之后 test data label的确定
        _, b_t_list_4_train, label_4_train = get_wave_frag(reservoir_data, processing_train = True)
        wave_frag_num, b_t_list = get_wave_frag(reservoir_data, processing_train = processing_train, test_seg_time = seg_Interval)
        # b_t_list_4_train, label_4_train 分别记录按照有标记的层段长度划分的储层 和 其对应的 label
        # label_4_test 分别是0和1对应的储层长度（时间表示）
        label_4_test = get_label_4_test(b_t_list_4_train, label_4_train, b_t_list)
    # print('wave_frag_num:',wave_frag_num)

    freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)  # 各分频率

    # 一段特征对应一个 label
    all_trace_data_feature = []
    all_lstm_label = []
    # all_trace_data 是同一口井 range 范围内的地震信息
    for trace_data in all_trace_data:
        if is_normalize:        # 如果数据需要进行归一化
            trace_data = normalize(trace_data)
        # 每个井的信息
        '''
        plt.plot(trace_data)
        plt.show()
        '''
        lstm_label = []
        reservoir_no = 0
        # /2 表示将时间表示改成第几个采样点(因为2ms是一个采样点)
        Cur_reservoir_bottom = int((b_t_list[0][0]) / 2)
        Cur_reservoir_top = int((b_t_list[0][1]) / 2)
        if feature_mode == 'spectrum':
            trace_data_feature = np.empty((wave_frag_num,int(fft_size/2)+1))   #当前 feature 长度是26
            Cur_feature = get_spectum(sampling_rate, trace_data[int(Cur_reservoir_bottom):int(Cur_reservoir_top)])
        elif feature_mode == 'origin':
            trace_data_feature = np.empty((wave_frag_num, math.ceil(seg_Interval/2)))
            Cur_feature = trace_data[int(Cur_reservoir_bottom):int(Cur_reservoir_top)]
            for _ in range(math.ceil(seg_Interval/2)-len(Cur_feature)):
                Cur_feature.extend([0])
        trace_data_feature[reservoir_no,:] = Cur_feature
        if processing_train:
            lstm_label.append(int(reservoir_data[0][3]))
        else:
            # 如果处理的是test data , lstm_label 需要 append每个储层对应的0的长度和1的长度
            pass
        reservoir_no += 1
        # 将 地震数据按照储层长度划分
        if processing_train:
            for reservoir in reservoir_data[1:]:
                iter_reservoir_bottom = float(reservoir[1])/2
                iter_reservoir_top = float(reservoir[2])/2
               # print('bottom:',iter_reservoir_bottom,'top:',iter_reservoir_top)
                # print(trace_data[iter_reservoir_bottom:iter_reservoir_top])

                if Cur_reservoir_top != iter_reservoir_bottom:
                    trace_data_feature[reservoir_no,:] = get_spectum(sampling_rate,trace_data[int(Cur_reservoir_top):int(iter_reservoir_bottom)])
                    reservoir_no += 1
                    lstm_label.append(0)
                trace_data_feature[reservoir_no,:] = get_spectum(sampling_rate,trace_data[int(iter_reservoir_bottom):int(iter_reservoir_top)])
                lstm_label.append(int(reservoir[3]))
                reservoir_no += 1
                Cur_reservoir_bottom = float(reservoir[1])/2
                Cur_reservoir_top = float(reservoir[2])/2
        # 将地震数据等间隔划分
        else:
            for reservoir in b_t_list[1:]:
                '''
                plt.plot(trace_data[int(reservoir[0]/2):int(reservoir[1]/2)])
                plt.xlabel(str(reservoir[0]/2) + '-' + str(reservoir[1]/2))
                plt.show()
                '''
                if feature_mode == 'spectrum':
                    Cur_feature = get_spectum(sampling_rate,trace_data[int(reservoir[0]/2):int(reservoir[1]/2)])
                elif feature_mode == 'origin':
                    Cur_feature = trace_data[int(reservoir[0]/2):int(reservoir[1]/2)]
                    for _ in range(math.ceil(seg_Interval / 2) - len(Cur_feature)):
                        Cur_feature.extend([0])
                trace_data_feature[reservoir_no,:] = Cur_feature
                reservoir_no += 1
            lstm_label = label_4_test
        all_trace_data_feature.append(trace_data_feature)
        all_lstm_label.append(lstm_label)
    # train 和 test 的数据 data feature 是一样的，但是lstm_label 不相同
    # train: [[0,0,0,1,1,1],[0,0,0,1,1,1]]
    # test:  [[[23,34],[45,12]],[[23,34],[45,12]]]  相加都等于57
    return all_trace_data_feature, all_lstm_label
def Cur_ts(reservoir_data,time_start,time_end):
    new_reservoir_data = []
    for reservoir_no in range(len(reservoir_data)):
        # 先找到timestart所在的层段
        if float(reservoir_data[reservoir_no][1]) < time_start and float(reservoir_data[reservoir_no][2]) < time_start:
            continue
        # 找到了time start 所在的层段
        elif float(reservoir_data[reservoir_no][1])<=time_start and float(reservoir_data[reservoir_no][2]) > time_start:
            first_reservoir = []
            first_reservoir.append(reservoir_data[reservoir_no][0]) # well_name
            first_reservoir.append(time_start)                      # reservoir_bottom
            first_reservoir.append(reservoir_data[reservoir_no][2]) # reservoir_top
            first_reservoir.append(reservoir_data[reservoir_no][3]) # reservoit_Info
            new_reservoir_data.append(first_reservoir)
        # 目标层段内的储层
        elif float(reservoir_data[reservoir_no][1])>time_start and float(reservoir_data[reservoir_no][2])>time_start and float(reservoir_data[reservoir_no][2])<time_end:
            new_reservoir_data.append(reservoir_data[reservoir_no])
        elif float(reservoir_data[reservoir_no][1])<time_end and float(reservoir_data[reservoir_no][2])>=time_end:
            new_reservoir_data.append([reservoir_data[reservoir_no][0],reservoir_data[reservoir_no][1],time_end,reservoir_data[reservoir_no][3]])
            return new_reservoir_data
def get_reservoir_data_for_data_aug(reservoir_data_receive, data_aug):
    '''
    根据滑动窗口生成不同的目标层段
    :param reservoir_data_receive: 专家划定的目标层段 ,里面的元素有 4 项，分别是 well_name , depth_bottom , depth_top , reservoir_Info，其中的储层顶和底都是用时间表示的
    :param data_aug: 是一个list, 表示滑动窗口的大小和滑动距离
    :return: reservoir_data_list : 表示生成的目标层段的 list
    '''
    reservoir_data_list = []
    # 表示一共可以生成多少个目标层段
    if int((reservoir_data_receive[-1][2]-reservoir_data_receive[0][1]-data_aug[0])/data_aug[1]) <=0:
        reservoir_data_list.append(reservoir_data_receive)
    else:
        for ts_no in range(int((reservoir_data_receive[-1][2]-reservoir_data_receive[0][1]-data_aug[0])/data_aug[1])):
            # time_start - time_end 为1000 ms
            time_start = float(reservoir_data_receive[0][1]) + ts_no*data_aug[1]
            time_end = time_start + data_aug[0]
            reservoir_data_list.append(Cur_ts(reservoir_data_receive,time_start,time_end))
    return reservoir_data_list
def get_label_equal_split(b_t_list_strict, label_strict, b_t_list):
    '''
    :param b_t_list_strict: 严格按照储层进行划分的时间顶底
    :param label_strict:    严格按照储层进行划分的label值
    :param b_t_list:        等分过后的顶底时间
    :return:                等分过后每个时间段0的长度和 1 的长度
    '''
    # 得到有 overlapping的  label，label中分别为 0 和 1 的长度
    label_equal_split = []
    for Cur_seg in b_t_list:        #等分后的时间顶底对
        # 分别找到Cur_seg的bottom和top位于哪个层段
        Cur_label = [0,0]
        for true_seg_no in range(len(b_t_list_strict)):
            if b_t_list_strict[true_seg_no][0] <= Cur_seg[0] and b_t_list_strict[true_seg_no][1] > Cur_seg[0]:  # 等分段的左点位于严格段的中间
                if Cur_seg[1]>b_t_list_strict[true_seg_no][1]: #等分段右点位于严格段右点的右边
                    Cur_label[label_strict[true_seg_no]] += (b_t_list_strict[true_seg_no][1] - Cur_seg[0])          # 等分段的label长度 + （严格段右点-等分段左点）
                else:
                    Cur_label[label_strict[true_seg_no]] += (Cur_seg[1]-Cur_seg[0])
            elif b_t_list_strict[true_seg_no][0] < Cur_seg[1] and b_t_list_strict[true_seg_no][1] >= Cur_seg[1]:
                Cur_label[label_strict[true_seg_no]] += (Cur_seg[1] - b_t_list_strict[true_seg_no][0])
                break
            elif b_t_list_strict[true_seg_no][0]>Cur_seg[0] and b_t_list_strict[true_seg_no][1]<Cur_seg[1]:
                Cur_label[label_strict[true_seg_no]] += (b_t_list_strict[true_seg_no][1]-b_t_list_strict[true_seg_no][0])
            else:
                continue
            #print(true_seg_no,Cur_label,b_t_list_4_train[true_seg_no],label_4_train[true_seg_no],'  ',b_t_list_4_train[true_seg_no][1]-b_t_list_4_train[true_seg_no][0])
        #exit()
        label_equal_split.append(Cur_label)
    return label_equal_split

def get_single_well_model_input(all_trace_data,reservoir_data_receive,feature_mode='spectrum',seg_Interval=50,
                                data_aug = None,is_normalize=False):
    '''
    :param all_trace_data:  list, 包含 当前well 对应的所有trace_data
    :param reservoir_data_receive:  里面的元素有 4 项，分别是 well_name , depth_bottom , depth_top , reservoir_Info，其中的储层顶和底都是用时间表示的
    :param data_aug: (list) 表示是否将数据进行扩充, 第一个元素表示滑动窗口大小，第二个元素表示滑动距离，第三个表示每个滑动窗口内各小窗口的重叠长度
    :param data_aug:            [x,y,z] ,x: 大时窗大小,如果为-1表示生成测试数据，大小需要根据顶底时间来确定, y ：大时窗重叠长度，z：小时窗重叠长度
    :return:
    '''
    # if feature_mode == 'spectum' 根据深度信息得到对应trace_data的频谱图
    # elif feature_mode == 'original' 返回原始振幅数据

    all_trace_data_feature_label_couple = []        # len = grid point, value:zip(feature,label)
    fft_size = sampling_rate
    # fft_size = int(seg_Interval/2)
    # wave_frag_num 表示被划分成几段 frag, b_t_list 表示当前井对应的 bottom 和 top 对
    # 如果是测试 data，首先生成 train data 的b_t_list 用于之后 test data label的确定，该函数用于构造不同的目标层段
    if data_aug[0] == -1:
        data_aug = [reservoir_data_receive[-1][2]-reservoir_data_receive[0][1],data_aug[1], data_aug[2]]
    reservoir_data_list = get_reservoir_data_for_data_aug(reservoir_data_receive, data_aug)

    '''
    print(len(reservoir_data_list[0]))
    for i in reservoir_data_list[0]:
        print(i)
    print('re_receive')
    for j in reservoir_data_receive:
        print(j)
    exit()
    '''
    for trace_data in all_trace_data:       # 共有 grid_points 个
        if is_normalize:
            trace_data = normalize(trace_data)
        Cur_trace_data_feature_label = []
        for reservoir_data in reservoir_data_list:
            Cur_trace_data_feature = []  # 用来保存每个储层段的特征
            # b_t_list_strict 是严格根据实际储层划分的顶底段，label_strict 是严格划分的顶底段对应的label值
            _, b_t_list_strict, label_strict = get_wave_frag(reservoir_data, mode='statistic')
            wave_frag_num, b_t_list = get_wave_frag(reservoir_data, mode='split_equally', seg_interval=seg_Interval,
                                                    overlapping=data_aug[2])
            # label_4_test 分别是0和1对应的储层长度（时间表示）
            label_equal_split = get_label_equal_split(b_t_list_strict, label_strict, b_t_list)
            '''
            print('b_t_list_strict: ',b_t_list_strict)
            print('label_strict:    ',label_strict)
            print('wave_frag_num:   ',wave_frag_num)
            print('b_t_list:         ',b_t_list)
            print('b_t_list:         ',len(b_t_list),[int(i[1]-i[0]) for i in b_t_list])
            print('label_equal_split:',len(label_equal_split),[int(sum(i)) for i in label_equal_split])
            exit()
            '''
            reservoir_no = 0
            # /2 表示将时间表示改成第几个采样点(因为2ms是一个采样点)
            Cur_reservoir_bottom = int((b_t_list[0][0]) / 2)
            Cur_reservoir_top = int((b_t_list[0][1]) / 2)
            if feature_mode == 'spectrum':
                trace_data_feature = np.empty((wave_frag_num, int(fft_size / 2) + 1))  # 当前 feature 长度是26
                Cur_feature = get_spectum(sampling_rate, trace_data[int(Cur_reservoir_bottom):int(Cur_reservoir_top)])
            elif feature_mode == 'origin':
                Cur_feature = trace_data[int(Cur_reservoir_bottom):int(Cur_reservoir_top)]
            # 放入第一个储层的信息
            Cur_trace_data_feature.append(Cur_feature)

            for reservoir in b_t_list[1:]:
                '''
                plt.plot(trace_data[int(reservoir[0]/2):int(reservoir[1]/2)])
                plt.xlabel(str(reservoir[0]/2) + '-' + str(reservoir[1]/2))
                plt.show()
                '''
                if feature_mode == 'spectrum':
                    Cur_feature = get_spectum(sampling_rate, trace_data[int(reservoir[0] / 2):int(reservoir[1] / 2)])
                elif feature_mode == 'origin':
                    Cur_feature = trace_data[int(reservoir[0] / 2):int(reservoir[1] / 2)]
                Cur_trace_data_feature.append(Cur_feature)
            # zip(Cur_trace_data_feature,label_equal_split --- > 当前300ms内所有的特征数据和label
            # depth_Info    ---> 当前300ms层段对应的深度信息（时间表示）
            depth_Info = [reservoir_data[0][1],reservoir_data[-1][2]]
            Cur_trace_data_feature_label.append([zip(Cur_trace_data_feature,label_equal_split),depth_Info])
        all_trace_data_feature_label_couple.append(Cur_trace_data_feature_label)
    return all_trace_data_feature_label_couple

def get_well_loc_name_dict():
    '''
    返回井位置和井名称对应的 dict
    :return:
    '''
    dict = {}
    well_loc_reader = csv.reader(open(file_loc_gl.well_loc_file))
    for row in well_loc_reader:
        if row[0] == 'well_no':
            continue
        dict.setdefault(str(int(float(row[2])))+','+str(int(float(row[1]))),row[0])
    return dict
def find_key(well_x,well_y,keys):
    if str(well_x)+','+str(well_y) in keys:
        return str(well_x)+','+str(well_y)
    else:
        offset_range = [-1,0,1]
        for x_offset in offset_range:
            for y_offset in offset_range:
                if str(well_x+x_offset)+','+str(well_y+y_offset) in keys:
                    return str(well_x+x_offset)+','+str(well_y+y_offset)

def get_well_name(well_x,well_y):
    '''
    根据井坐标得到 井名称
    :param well_x:  int
    :param well_y:  int
    :return:
    '''
    # key 是井坐标，value 是井名称
    well_loc_name_dict = get_well_loc_name_dict()
    key = find_key(well_x,well_y,well_loc_name_dict.keys())
    return well_loc_name_dict.get(key)

def get_model_feature_label(all_wells_trace_data,well_reservoir_file,depth_time_rel_dir,feature_mode='spectum',
                            seg_Interval=50, data_aug = None,is_normalize=False):
    '''
    :param all_wells_trace_data: dict, keys are the location of all wells , values are their trace_data
    :param well_reservoir_reader: csv.reader including all reaervoir data of all wells
    :param depth_time_rel_dir:  dir of saving all depth_time_rels of all wells
    :param data_aug:            [x,y,z] ,x: 大时窗大小,如果为-1表示生成测试数据，大小需要根据顶底时间来确定, y ：大时窗重叠长度，z：小时窗重叠长度
    :return:
        lstm_input :list, spectum of all wells
        lstm_label :list, label seqs of each well
    '''
    all_wells_feature_label_zip = []
    well_processed_num = 1
    lost_well_num = 0
    # key 是 110口井的 key, 在 井的储层信息中可能有些 key 不存在, 6....,4....
    for key in sorted(all_wells_trace_data.keys()):

        well_name = get_well_name(int(float(key.split(',')[1])), int(float(key.split(',')[0])))

        well_processed_num += 1
        # 先得到当前well_name对应的储层信息 (时间表示) 储层元素value包括四部分，well_name，reservoir_bottom，reservoir_top，是不是储层
        # well_reservoir_reader_filepath 是目标层段的信息
        # 该 dict 是根据 每个井对应的储层信息得到的
        well_reservoir_dict = get_well_reservoir_dict(well_reservoir_file)

        # 没有找到与当前井的名称对应的储层信息
        if well_reservoir_dict.get(well_name) == None:
            #print('没有找到与当前井的名称对应的储层信息:',well_name)
            lost_well_num +=1
            #print(well_name)
            continue
        print('正在生成井：%s的训练数据(%g/%g,grid points=%g)'%(well_name,well_processed_num,len(all_wells_trace_data.keys()),len(all_wells_trace_data.get(key))))
        # 得到的Cur_well_reservoir_data 是时间数据,即时间段 - （0/1）
        Cur_well_reservoir_data = get_specific_reservoir_data(well_reservoir_dict.get(well_name),depth_time_rel_dir,well_name)

        if Cur_well_reservoir_data == []:
            print('得到一个空的List')
            exit(0)
        # 根据trace_data 和 Cur_well_reservoir_data 得到当前well的model feature
        # model_label 是一个list，里面分别存放的是非储层长度和储层长度
        # all_wells_trace_data.get(key) 表示当前井对应的所有 trace data 范围是reservoir_range
        # 如果使用了 data_aug 参数，则生成的Cur_well_feature 和 Cur_well_label 是经过滑动窗口之后的，要注意append的时候范围

        if data_aug is None:
            Cur_well_feature , Cur_well_label = get_single_well_lstm_input(all_wells_trace_data.get(key),
                                                                           Cur_well_reservoir_data,feature_mode,
                                                                           seg_Interval=seg_Interval, data_aug = data_aug,
                                                                           is_normalize=is_normalize)
        # 如果使用大时窗进行数据扩充
        else:
            single_feature_label_zip_list = get_single_well_model_input(all_wells_trace_data.get(key),
                                                                        Cur_well_reservoir_data,feature_mode,
                                                                        seg_Interval=seg_Interval, data_aug = data_aug,
                                                                        is_normalize=is_normalize)
        all_wells_feature_label_zip.append([well_name,single_feature_label_zip_list])

    return all_wells_feature_label_zip

def get_trace_data(trace_data_file):
    with open(trace_data_file,'rb') as file:
        reservoir_range = pickle.load(file)
        all_trace_data_dict = pickle.load(file)
    return reservoir_range, all_trace_data_dict
def is_high_correlation(feature_file, attr_name, correlation_data):
    print(feature_file)
    for key in correlation_data.keys():
        #print(feature_file)
        if feature_file[feature_file.index(attr_name)+len(attr_name)+1:] in key:
            return True
    return False
def save_feature_data(use_target_segment = True, seg_Interval = 50,feature_type = 'spectrum', data_aug = None,is_normalize = False):
    '''
    :param use_target_segment:  是否使用目标层段
    :param seg_Interval:        小时窗的大小  25/50
    :param feature_type:        生成的特征类型     'spectrum' / 'origin'
    :param input_dim:           生成的训练数据输入维度的大小
    :param data_aug:            [x,y,z] ,x: 大时窗大小,如果为-1表示生成测试数据，大小需要根据顶底时间来确定, y ：大时窗重叠长度，z：小时窗重叠长度
    :return:
    '''
    correlation_data = read_high_correlation_files()
    trace_data_file_Base = file_loc_gl.full_train_data
    if data_aug[0] != -1:
        first_step_dir = '4-training_data'
    else:
        first_step_dir = '5-testing_data'
    for child_dir in os.listdir(trace_data_file_Base):
        #if child_dir == 'frequencyseis' or child_dir == 'phaseseis':
        #    continue
        trace_dada_dir = os.path.join(trace_data_file_Base,child_dir)
        if not os.path.isdir(trace_dada_dir):
            continue
        for feature_file in os.listdir(trace_dada_dir):
            # 只选择高相关性的feature
            print(is_high_correlation(feature_file,child_dir,correlation_data))
            if not is_high_correlation(feature_file,child_dir,correlation_data):
                print('文件：%s 相关性较低'%feature_file)
                continue

            # 每个属性文件的地址
            if '.pkl' not in feature_file:
                continue
            attr_file_path = os.path.join(trace_dada_dir,feature_file)
            if feature_type == 'origin':
                input_dim = math.ceil(seg_Interval/2)

            # 保存特征文件的地址
            data_saved_path = []
            saved_dir = os.path.join('data',first_step_dir,'normalize_'+str(is_normalize),
                                     feature_type,'seg_Interval_'+str(seg_Interval), child_dir)
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)

            if not use_target_segment:
                train_data_saved_path = os.path.join(saved_dir,feature_file + '.train')
            else:
                train_data_saved_path = os.path.join(saved_dir,feature_file + '_ts.train')
            if not data_aug is None:
                train_data_saved_path = train_data_saved_path[:-6]+'_aug_'+str(data_aug[0])+'_'+str(data_aug[1])+'_overlapping.train.pkl'

            if not os.path.exists(train_data_saved_path):
                print('正在处理：',child_dir,' - ',feature_file)
                # 得到的 trace_data 是一个dict，key为每口井的坐标（String），value为井坐标对应的trace_data,是一个list
                reservoir_range, all_wells_trace_data = get_trace_data(attr_file_path)
                # 储层数据
                well_reservoir_file_path = file_loc_gl.well_post_data
                if not use_target_segment:
                    well_reservoir_file_name = 'well_reservoir_rock_oil_merged.csv'
                else:
                    # well_reservoir_file_name = 'well_reservoir_rock_oil_merged_target_segment.csv'
                    well_reservoir_file_name = 'well_reservoir_Info_clean.csv'

                # 时深关系文件夹
                depth_time_rel_dir = file_loc_gl.depth_time_rel_dir
                well_reservoir_file = os.path.join(well_reservoir_file_path,well_reservoir_file_name)
                    # 得到所有训练数据
                    # feature_mode : 表示特征模式。 spectum：使用频谱图，original：使用原始数据
                # data_aug 表示是否进行data_augmentation
                train_feature_label = get_model_feature_label(all_wells_trace_data,well_reservoir_file,depth_time_rel_dir,
                                                              feature_mode=feature_type,seg_Interval = seg_Interval,
                                                              data_aug=data_aug,is_normalize=is_normalize)


                # save_all_training_data
                # 生成的文件格式：
                # train_feature_label = [每个井数据扩充之后的数据] × 井数量
                # [每个井数据扩充之后的数据] = [每个井range范围内每个trace扩充之后的数据] × grid point
                # [每个井range范围内每个trace扩充之后的数据] = [每个300ms对应的储层标记信息] × 300ms 大时窗的个数
                # [每个300ms对应的储层标记信息] = [zip(储层段特征list,对应的标记list),[深度顶，深度底]]
                with open(train_data_saved_path,'wb') as file:
                    pickle.dump(train_feature_label,file,-1)
            else:
                print('特征文件：',train_data_saved_path,'已存在!')


def save_all_features(data_aug=None,seg_Intervals = [30],is_normalize = False):
    '''
    :param data_aug:
    :return:
    '''
    global sampling_rate
    sampling_rate = 160
    input_dim = int(sampling_rate / 2 + 1)  # 表示使用 ‘spectrum’特征时，与采样率对应的输入维度
    # feature_types = ['origin','spectrum']   # 表示特征类型
    feature_types = ['origin']
    is_use_target_segments = [True]         # 表示是否使用目标层段
    seg_Intervals = seg_Intervals                 # 表示小时窗的大小
    for feature_type in feature_types:
        for seg_Interval in seg_Intervals:
            for is_use_target_segment in is_use_target_segments:
                print(feature_type,seg_Interval,is_use_target_segment)
                # 把所有的波形数据变成频谱特征并保存下来
                # seg_Interval 表示划分的间隔（时间表示）
                # train_part 表示训练数据的比例，训练数据存入 train_files,测试数据存入test_files
                save_feature_data(use_target_segment=is_use_target_segment, seg_Interval=seg_Interval,
                                  feature_type=feature_type,data_aug=data_aug,is_normalize=is_normalize)
def show_feature_data_structure(sourceFile):
    #sourceFile = '../../data/4-training_data/origin/seg_Interval_25/ampseimic/Trace_data_around_wells_range_3_ampseimic_EnvelopeDerivative_100hz.pkl_ts_aug_300_2_overlapping.train.pkl'
    #sourceFile = '../../data/5-testing_data/origin/seg_Interval_30/ampseimic/Trace_data_around_wells_range_3_ampseimic_SignalEnvelope_50hz.pkl_ts_aug_-1_2_overlapping.train.pkl'

    with open(sourceFile,'rb') as file:
        data = pickle.load(file)
        print('共有%g口井'%len(data))
        well_no = input('显示第几口井：')
        print('well name:%s, grid points = %g'%(data[int(well_no)][0],len(data[int(well_no)][1])))
        print('当前井共划分大时窗%g个'%len(data[int(well_no)][1][0])) # 0 表示 grid 中的第一个trace
        big_windows = input('请输入要显示第几个大时窗')
        print('     第%s个大时窗的信息如下：'%big_windows)

        print(data[int(well_no)][1][0][int(big_windows)])
        print('     深度：%g(ms) - %g(ms)'%(data[int(well_no)][1][0][int(big_windows)][1][0],data[int(well_no)][1][0][int(big_windows)][1][1]))
        for i in data[int(well_no)][1][0][int(big_windows)][0]:
            print('     input_dim:  %g,   t:%g (ms)'%(len(i[0]),sum(i[1])))

        #print('     每个储层段的纬度：')
        #for i in data[int(well_no)][1][0][int(big_windows)][0]:
        #    print('     ', len(i))
if __name__ == '__main__':
    # 保存所有数据的特征
    # data_aug 里面的参数表示做 data_augmentation 的时候窗口大小(ms)滑动步长(ms),第三个元素表示每个滑动窗口内各小滑动窗口的重叠长度
    # 生成的文件格式：
    # train_feature_label = [well_name,每个井数据扩充之后的数据] × 井数量
    # [每个井数据扩充之后的数据] = [每个井range范围内每个trace扩充之后的数据] × grid point     (grid的排列方式是从左下向右向上)
    # [每个井range范围内每个trace扩充之后的数据] = [每个300ms对应的储层标记信息] × 300ms 大时窗的个数
    # [每个300ms对应的储层标记信息] = [zip(储层段特征list,对应的标记list),[深度顶，深度底]]
    # 对于测试数据大时窗的大小应该就取目标层段的长度，在下面的函数中设为-1，大小由井的最顶和最底的差来决定,但是小时窗的大小和每个小时窗之间
    # 的重叠长度依然与训练数据的一致
    save_all_features(data_aug=[310,2,10])     # 使用300毫秒的大时窗进行数据的扩充
    save_all_features(data_aug=[-1,2,10])       # 不使用大时窗进行划分
    show_feature_data_structure()   # 显示数据的结构