"""
# 生成某个属性文件中，某个井对应的地震波曲线
"""
import os
import struct
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from DLmodel.get_trace_data_around_wells import Build_Interpolation_function
from scipy import interpolate


def get_well_loc_name_dict():
    '''
    返回井位置和井名称对应的 dict
    :return:
    '''
    dict = {}
    well_loc_reader = csv.reader(open('..\\..\\data\\3-分析数据\\well_location_new.csv'))
    for row in well_loc_reader:
        if row[0] == 'well_no':
            continue
        dict.setdefault(str(int(float(row[2]))) + ',' + str(int(float(row[1]))), row[0])
    return dict


def find_key(well_x, well_y, keys):
    if str(well_x) + ',' + str(well_y) in keys:
        return str(well_x) + ',' + str(well_y)
    else:
        offset_range = [-1, 0, 1]
        for x_offset in offset_range:
            for y_offset in offset_range:
                if str(well_x + x_offset) + ',' + str(well_y + y_offset) in keys:
                    return str(well_x + x_offset) + ',' + str(well_y + y_offset)


def get_well_name(well_x, well_y):
    '''
    根据井坐标得到 井名称
    :param well_x:  int
    :param well_y:  int
    :return:
    '''
    well_loc_name_dict = get_well_loc_name_dict()
    key = find_key(well_x, well_y, well_loc_name_dict.keys())
    return well_loc_name_dict.get(key)


def get_trace_data(trace_data_file):
    # 支持 range = 任意的情况
    all_wells_trace_data = {}
    with open(trace_data_file, 'rb') as f:
        global reservoir_range
        reservoir_range = int(struct.unpack('!f', f.read(4))[0])
        print('reservoir_range:', reservoir_range)
        # 得到井的数目
        # (all_bytes - 4[re_range])/(well_x + well_y + 1251*4*(1+2*reservoir_range)*(1+2*reservoir_range))
        # well_num = int((os.path.getsize(trace_data_file)-4)/(1251*4+8))
        well_num = int((os.path.getsize(trace_data_file) - 4) / (
        1251 * 4 * (1 + 2 * reservoir_range) * (1 + 2 * reservoir_range) + 8))
        for well_no in range(well_num):
            well_x = struct.unpack('!f', f.read(4))[0]
            well_y = struct.unpack('!f', f.read(4))[0]
            Cur_well_data = []
            for trace_no in range((1 + 2 * reservoir_range) * (1 + 2 * reservoir_range)):
                Cur_trace_data = []
                for amp_point in range(1251):
                    Cur_trace_data.append(struct.unpack('!f', f.read(4))[0])
                Cur_well_data.append(Cur_trace_data)
            Cur_key = str(well_x) + ',' + str(well_y)
            all_wells_trace_data.setdefault(Cur_key, Cur_well_data)
    return all_wells_trace_data


def Build_Interpolation_function(depth_time_rel_filepath, is_reverse=False):
    # depth_time_rel_filepath = 'H:\\研究生\\2-胜利油田\\数据\\1-well\\depth_time_rel\\cb6_ck.txt'
    rel_file = open(depth_time_rel_filepath, 'r')
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
    while (Cur_line):
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

    # f = interpolate.interp1d(depth_list, time_list, kind=kind[selection_number]) # 三阶样条插值
    if not is_reverse:
        f = interpolate.UnivariateSpline(depth_list, time_list, k=1, s=2)  # 三阶样条插值
    elif is_reverse:
        f = interpolate.UnivariateSpline(time_list, depth_list, k=1, s=2)  # 三阶样条插值
    rel_file.close()
    return depth_list, time_list, f


def get_oil_attr(well_name):
    import csv
    file_path = 'E:\\Programming\\Python_workspace\\Shengli\\data\\1-well\\oil_data\\' + well_name + '.csv'
    bottom_list = []
    top_list = []
    label_list = []
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            if line[0] == '井号':
                continue
            else:
                if line[6] == '油层':
                    Cur_label = 2
                elif line[6] == '水层':
                    Cur_label = 3
                elif line[6] == '油水同层':
                    Cur_label = 4
                else:
                    continue
                bottom_list.append(float(line[2]))
                top_list.append(float(line[3]))
                label_list.append(Cur_label)
    return bottom_list, top_list, label_list


def get_rock_attr(well_name):
    import csv
    file_path = 'E:\\Programming\\Python_workspace\\Shengli\\data\\1-well\\post_data\\well_reservior_rock_data.csv'
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        bottom_list = []
        top_list = []
        label_list = []
        for line in csv_reader:
            if not line[0] == well_name:
                continue
            else:
                bottom_list.append(float(line[1]))
                top_list.append(float(line[2]))
                label_list.append(int(line[3]))
    return bottom_list, top_list, label_list


def draw_pic(bottom_list, top_list, label_list):
    pass


def main():
    # 生成某个属性文件中，某个井对应的地震波曲线

    source_file = 'E:\\Programming\\Python_workspace\\Shengli\\data\\3-分析数据\\seismic\\All_trace_data_around_wells_range_0_seismic_CDD_bigdata_from_petrel.sgy'
    trace_data_list = get_trace_data(source_file)
    # CB30
    well_name_see = 'SHHG1'
    depth_time_rel_filepath = '..\\..\\data\\1-well\\depth_time_rel\\' + well_name_see + '_ck.txt'
    depth_list, time_list, interpolation_f = Build_Interpolation_function(depth_time_rel_filepath)
    # depth_list_t = [int(i) for i in map(interpolation_f,depth_list)]

    bottom_list, top_list, label_list = get_oil_attr(well_name_see)
    print(bottom_list)
    print(top_list)
    print(label_list)
    bottom_list = [int(i) for i in map(interpolation_f, bottom_list)]
    top_list = [int(i) for i in map(interpolation_f, top_list)]
    rock_bottom_list, rock_top_list, rock_label_list = get_rock_attr(well_name_see)
    rock_bottom_list = [int(i) for i in map(interpolation_f, rock_bottom_list)]
    rock_top_list = [int(i) for i in map(interpolation_f, rock_top_list)]
    bottom_list.extend(rock_bottom_list)
    top_list.extend(rock_top_list)
    label_list.extend(rock_label_list)
    print(bottom_list)
    print(top_list)
    print(label_list)
    draw_pic(bottom_list, top_list, label_list)
    for key in sorted(trace_data_list.keys()):
        well_name = get_well_name(int(float(key.split(',')[0])), int(float(key.split(',')[1])))
        print('well_name:', well_name)
        if well_name == well_name_see:
            print(trace_data_list.get(key)[0])
            plt.plot(trace_data_list.get(key)[0])
            plt.show()
            break
        else:
            continue


if __name__ == '__main__':
    main()