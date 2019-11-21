import os
import pickle
import struct

cdp_s = 1189
cdp_e = 1852
line_s = 627
line_e = 2267
sampling_points = 1251
total_trace = (line_e-line_s+1)*(cdp_e-cdp_s+1)

def get_point_pred_data(line, cdp, time, seismic_file):
    """
    返回对应 line， cdp，time坐标点的pred
    :param line:
    :param cdp:
    :param time: 已经转化成采样点表示
    :param seismic_file: 打开的sgy文件
    :return:
    """

    # get point
    index = (line - line_s) * (cdp_e - cdp_s + 1) + (cdp - cdp_s)
    ipoint = 3840 + index * (sampling_points * 4 + 240)
    with open(seismic_file,'rb') as file:
        file.seek(0)
        file.seek(ipoint + time * 4)  # 表示从文件头开始移动文件指针
        value = struct.unpack('!f', file.read(4))[0]

    return value


sgy_file = "result/cnn_target/cnn_pred_petrel_Time_gain_attr_test.sgy"
print(get_point_pred_data(1431, 1394, 912, sgy_file))