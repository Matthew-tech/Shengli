"""
author: Eric
update time:2017.07.26
Description: extract trace data from seismic cube
             generate pickle file:
             reservoir_range
             {well_loc:trace_data_list}
"""
import csv
import struct
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from decimal import getcontext, Decimal
import pickle
from Configure.global_config import *
files = file_loc_global()
def print2csv(mode,rate,content=None):
    with open(os.path.join(file_loc_gl.infopresent,mode+'_info.csv'),'w') as file:
        writer = csv.writer(file,lineterminator='\n')
        writer.writerow([content,rate])
def get_well_cdp_incline(well_x, well_y):
    well_line = int(round((well_x - region_x_s) / delta_x)) + line_s
    well_cdp = int(round((well_y - region_y_s) / delta_y)) + cdp_s
    return well_cdp, well_line

def get_grid_trace_data(well_cdp_center,well_line_center,seismic_file,return_type = 'list'):
    reservoir_range = 3
    trace_data = []
    reservoir_range_list = []
    reservoir_range_list.extend(list(reversed(range(reservoir_range + 1))))
    reservoir_range_list = [i * (-1) for i in reservoir_range_list]
    reservoir_range_list.extend(list(range(reservoir_range + 1)))
    del reservoir_range_list[reservoir_range]
    # 将文件指针移动到当前grid 对应的坐标原点
    well_line_origin = well_line_center+reservoir_range_list[0]
    well_cdp_origin = well_cdp_center + reservoir_range_list[0]
    seismic_file.seek(0)
    volumn_head = seismic_file.read(3600)                   #每个地震体的卷头
    # well_cdp,well_incline = get_well_cdp_incline(well_x,well_y)
    if well_line_origin - line_s >0 and well_cdp_origin - cdp_s >0:
        seismic_file.read((well_line_origin - line_s) * (cdp_e - cdp_s + 1) * (sampling_points * 4 + 240))      # 先跳过前面所有的line
        seismic_file.read((well_cdp_origin - cdp_s) * (sampling_points * 4 + 240))   # 跳过前面的cdp
        reservoir_range_list_cdp = reservoir_range_list.copy()
    else: # 表示超出范围,只取一半
        well_line_origin = well_line_center + reservoir_range_list[0]
        well_cdp_origin = well_cdp_center
        seismic_file.read((well_line_origin - line_s) * (cdp_e - cdp_s + 1) * (sampling_points * 4 + 240))  # 先跳过前面所有的line
        # seismic_file.read((well_cdp_origin - cdp_s) * (sampling_points * 4 + 240))  # 跳过前面的cdp
        reservoir_range_list_cdp = reservoir_range_list[int(len(reservoir_range_list)/2):]
    # 先读取最下面line的几个cdp对应的trace data
    for _ in reservoir_range_list:  # 遍历所有的线号
        for _ in reservoir_range_list_cdp: # 遍历所有的道号 读取当前line对应的道号
            seismic_file.read(240)
            Cur_trace_data = []
            for point_i in range(sampling_points):
                Cur_trace_data.append(struct.unpack('!f', seismic_file.read(4))[0])
            trace_data.append(Cur_trace_data)
        # 跳过当前line后面所有的cdp 和 上一个line前面所有的cdp(一共(cdp_e - cdp_s)-len(reservoir_range_list)个)
        seismic_file.read(((cdp_e - cdp_s+1)-len(reservoir_range_list)) * (sampling_points * 4 + 240))
    return trace_data
def get_trace_data_around_single_well(well_x, well_y, seismic_sgy_file, reservoir_range=0,return_type='list'):
    well_cdp_center, well_line_center = get_well_cdp_incline(well_x, well_y)
    # list of trace
    single_well_trace_data = get_grid_trace_data(well_cdp_center,well_line_center,seismic_sgy_file,return_type=return_type)
    return single_well_trace_data


def get_trace_data_around_all_wells(well_loc_file_path, seismic_sgy_file, saveFilePath='', reservoir_range=0,
                                    return_type='list'):
    '''
    :param well_loc_file_path:  包含井的位置坐标
    :param seismic_sgy_file:    地震体文件
    :param saveFilePath:        存放地址
    :param reservoir_range:
    :param return_type:         list 表示将其存入pkl文件
    :return:
    '''
    with open(well_loc_file_path, 'r') as well_loc_file:
        with open(saveFilePath, 'wb') as saveFile:
            pickle.dump(reservoir_range,saveFile)   # 存入井周围的大小range
            well_loc_file.readline()
            Cur_well = well_loc_file.readline()
            well_processed = 1
            well_loc_trace_dict = {}
            well_location_all = []
            # 遍历每一口井,将坐标存到 list 中，并先按照x进行排序，再按照 y 进行排序
            while (Cur_well):
                Cur_well = Cur_well.split(',')
                well_name = Cur_well[0]
                # 井坐标
                well_x = float(Cur_well[1])     # 对应 line
                well_y = float(Cur_well[2])     # 对应 cdp
                well_location_all.append([well_name,well_x,well_y])
                Cur_well = well_loc_file.readline()
            # 对井的坐标进行排序
            well_location_all.sort(key=lambda x:(x[1],x[2]))

            for well_no in range(len(well_location_all)):
                well_name = well_location_all[well_no][0]
                well_x = well_location_all[well_no][1]
                well_y = well_location_all[well_no][2]
                print('正在获取井(', well_name, ')附近', 'range=', reservoir_range, '的地震波数据...', well_processed)
                # trace_data 是一个list，里面的元素为 well_x，well_y 附近的 reservoir_range 范围内的数据
                trace_data = get_trace_data_around_single_well(well_x, well_y, seismic_sgy_file,
                                                               reservoir_range=reservoir_range, return_type=return_type)
                # 存入当前trace_data 对应的井的x，y坐标,然后再写入 对应的trace_data
                well_loc_trace_dict.update({str(well_x) + ',' + str(well_y): trace_data})
                well_processed += 1
            pickle.dump(well_loc_trace_dict,saveFile,-1)


def get_well_time_reservoir(reservoir_file, well_name, interpolation_f):
    time_top_list = []
    time_bottom_list = []
    reservoir_Info = []
    reservoir_file.readline()
    print('正在进行', well_name, '的时深转换...')
    reservoir_num = 0
    reservoir_start = 1
    reservoir_end = 515
    for reservoir_i in range(reservoir_start - 1):
        reservoir_file.readline()
    while (reservoir_num < (reservoir_end - reservoir_start + 1)):
        CurLine = reservoir_file.readline()
        if isinstance(CurLine, str):
            CurLine = CurLine.split(',')
            reservoir_num += 1
            time_top_list.append(interpolation_f(float(CurLine[1])))
            time_bottom_list.append(interpolation_f(float(CurLine[2])))
            reservoir_Info.append(CurLine[3])
    return time_top_list, time_bottom_list, reservoir_Info


def draw_interpolation_curve(depth_list, time_list, interpolation_f):
    new_depth = list(np.linspace(0, max(depth_list), 10000))
    plt.plot(depth_list, time_list, 'r')
    plt.plot(new_depth, interpolation_f(new_depth), 'g')
    plt.show()


def draw_reservoir_Info(time_top_list, time_bottom_list, reservoir_Info):
    for reservoir in range(len(time_bottom_list)):
        reservoir_start = time_top_list[reservoir]
        reservoir_end = time_bottom_list[reservoir]
        plt.plot(list(np.linspace(reservoir_start, reservoir_end, 20)), [100] * 20, 'black')
    plt.show()
def plot_trace_data(sourceFile):
    with open(sourceFile,'rb') as file:
        reservoir_range = pickle.load(file)
        all_trace_data = pickle.load(file)
        print(len(all_trace_data.keys()))
        for key in sorted(all_trace_data.keys()):
            plt.plot(all_trace_data.get(key)[0])
            plt.show()

def start_extractting():
    well_loc_file = files.well_loc_file
    seismic_sgy_file_path_base = files.seismic_sgy_file_path_base
    for reservoir_range in range(3, 4):
        for child_dir in os.listdir(seismic_sgy_file_path_base):
            feaature_file_dir = os.path.join(seismic_sgy_file_path_base, child_dir)
            for file_name in os.listdir(feaature_file_dir):
                file_path = os.path.join(feaature_file_dir, file_name)
                print('正在提取：', file_path, 'trace data')
                with open(file_path, 'rb') as seismic_sgy_file:
                    saveFilePath_Base = os.path.join(files.saveFilePath_Base, child_dir)
                    saveFile_Name = 'Trace_data_around_wells_range_' + str(
                        reservoir_range) + '_' + child_dir + '_' + file_name + '.pkl'
                    if not os.path.exists(saveFilePath_Base):
                        os.makedirs(saveFilePath_Base)
                    if not os.path.exists(os.path.join(saveFilePath_Base, saveFile_Name)):
                        # return_type = list ---> 存入pkl文件
                        # return_type = binary ---> 使用4字节方式进行存储
                        get_trace_data_around_all_wells(well_loc_file, seismic_sgy_file,
                                                        os.path.join(saveFilePath_Base, saveFile_Name)
                                                        , reservoir_range=reservoir_range, return_type='list')
                        # plot_trace_data(os.path.join(saveFilePath_Base, saveFile_Name))

if __name__ == '__main__':

    start_extractting()
