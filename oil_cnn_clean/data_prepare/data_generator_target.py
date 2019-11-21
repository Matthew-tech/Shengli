# -*- coding:utf-8 -*-
# time:2018/6/7 下午1:38
# author:ZhaoH

import pickle
import struct
import numpy as np
import os


cdp_s = 1189
cdp_e = 1852
line_s = 627
line_e = 2267
sampling_points = 1251
total_trace = (line_e-line_s+1)*(cdp_e-cdp_s+1)
ten_line_total_trace = 10*(cdp_e-cdp_s+1)
slice_step = 14


def get_slice_data(line, cdp, time, seismic_file):  # time is sample number
    """
    返回对应 line， cdp，time坐标点的image
    :param line:
    :param cdp:
    :param time: 已经转化成采样点表示
    :param seismic_file: 打开的sgy文件
    :return: 返回三个切面的array
    """
    tmpdata = []
    step = int(slice_step / 2)
    # get time slice
    for iline in range(line - step, line + step):
        for icdp in range(cdp - step, cdp + step):
            index = (iline - line_s) * (cdp_e - cdp_s + 1) + (icdp - cdp_s)
            if (index >= 0 and index < total_trace):
                ipoint = 3840 + index * (sampling_points * 4 + 240)
                seismic_file.seek(ipoint + time * 4)                    # 表示从文件头开始移动文件指针
                value = struct.unpack('!f', seismic_file.read(4))[0]
                tmpdata.append(value)
            else:
                tmpdata.append(0)
    image = np.asarray(tmpdata).reshape((slice_step,slice_step, 1))

    return image


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


def get_target_horizon_slice(line_s, line_e, loc_time_dict_t0, loc_time_dict_t11, sgy_info=[]):
    """
        获得目标层段间10个line的测试数据
        :param line_s:  预测的第一条line
        :param line_e:  预测的最后一条line
        :param time_line_cdp:  line_cdp对应的time
        :param sgy_info:  [sgy_dir, sgy filename]
        :return:
        """
    sgyfile = os.path.join(sgy_info[0], sgy_info[1])

    horizon_data_dir = 'data/cnn_target/'
    if not os.path.exists(horizon_data_dir): os.makedirs(horizon_data_dir)
    ten_line_data_saved = horizon_data_dir + "line_{}_{}.ht".format(line_s, line_e)
    ten_line_data = []
    if not os.path.exists(ten_line_data_saved):
        with open(sgyfile, 'rb') as file:
            count = 0
            for Cur_line in range(line_s, line_e+1):
                for Cur_cdp in range(cdp_s + 1, cdp_e - 1 + 1):
                    key = "{}-{}".format(Cur_line, Cur_cdp)
                    if key in loc_time_dict_t0:
                        if key in loc_time_dict_t11:
                            cur_time_t0 = int(loc_time_dict_t0.get(key) / 2)
                            #print(loc_time_dict_t11.get(key))
                            cur_time_t11 = int(loc_time_dict_t11.get(key) / 2)
                            for Cur_time in range(cur_time_t0, cur_time_t11+1):
                                Cur_data = get_slice_data(line=Cur_line, cdp=Cur_cdp, time=Cur_time, seismic_file=file)
                                ten_line_data.append(Cur_data)
                        else:
                            #print(key)
                            ten_line_data.append(np.empty(shape=(slice_step, slice_step, 1)))
                    else:
                        #print(key)
                        ten_line_data.append(np.empty(shape=(slice_step, slice_step, 1)))
                    count += 1
                    if count % 1000 == 0: print("finish {}".format(count / ten_line_total_trace))

        with open(ten_line_data_saved, "wb") as file_saved:
            pickle.dump(ten_line_data, file_saved, -1)
    else:
        print(ten_line_data_saved + 'finished!')
        with open(ten_line_data_saved, 'rb') as file_exist:
            ten_line_data = pickle.load(file_exist)

    return ten_line_data


if __name__ == '__main__':
    # train_model(2)
    # sgy_info = ["data/seismic", "petrel_Time_gain_attr.sgy"]
    get_slice_data(line, cdp, time, seismic_file)
    # pred_time(2)