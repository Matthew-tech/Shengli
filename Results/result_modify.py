#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-9 下午12:15
# @Author  : Eric
# @File    : result_modify.py

# 修改生成的Result文件，把前两个线进行填充
import os
from Configure.global_config import cdp_s, cdp_e, line_s, line_e, file_loc_gl,sampling_points
import numpy as np
import sys
import struct
#from models.DLmodel.prediction.point_to_label.label_all_data_birnn import get_input_len
line_skip = 0
attr_num = 76
cdp_num = cdp_e - cdp_s + 1  # 表示每个线的道数
line_num = line_e - line_s + 1  # 一共这么多线
read_num = 664  # 表示一次读取100个道

def add_info(file, trace_head_all):
    """
    将头信息加入file，并将剩下的1251个采样点充0
    :param file:  文件指针
    :param trace_head: 包含 trace_head 的一个list
    :return:
    """
    for trace_head in trace_head_all:
        file.write(trace_head)
        for _ in range(sampling_points):
            file.write(struct.pack('!f', float(0)))


def get_trace_head(from_file, trace_no, read_num):
    """
    从 from_file 中获取道头的信息，并将其返回
    :param from_file:   读取道头信息的文件路径
    :param trace_no:    表示从 trace_no 开始进行读取
    :param read_num:    一共读取read_num 的道，并将其返回
    :return:  list 格式
    """
    with open(from_file, 'rb') as file:
        volumn_head = file.read(3600)
        skip_bytes = trace_no * (240 + 1251 * 4)
        file.seek(skip_bytes, 1)
        # 开始读取 read_num 个数的道头信息
        trace_head_all = []
        for _ in range(read_num):
            trace_head_all.append(file.read(240))
            for _ in range(sampling_points):
                file.read(4)
        return trace_head_all


def add_trace_head(filepath='',last=True):
    """
    对sourcedir 中的地震结果进行修改
    :param sourcefile:
    :param last:
    :return:
    """
    result_mod = filepath + '_mod.sgy'
    # 打开文件，并将前两条线的道头进行写入
    with open(filepath, 'rb') as file1, open(result_mod, 'wb') as file2:
        print('正在写入头部信息')
        volumn_head = file1.read(3600)
        file2.write(volumn_head)
        # file2 中写入边缘的两条线
        print('正在添加前2线的信息...')
        for trace_no in np.arange(0, 2 * cdp_num, read_num):  # 遍历文件中的每一道，trace_no 从 0 开始计数
            print('%g - %g' % (trace_no, trace_no + read_num))
            # 获取从trace_no 开始 read_num 个数的道头信息
            from_file = os.path.join(file_loc_gl.seismic_sgy_file_path_base,'seismic/CDD_bigdata_from_petrel.sgy')
            trace_head = get_trace_head(from_file, trace_no, read_num)
            print('道数：%g' % len(trace_head))
            add_info(file2, trace_head)
        # 将 file1 中剩下的信息加入 file2 当中
        print('正在写入剩余的信息')
        last_bytes = file1.read()
        file2.write(last_bytes)
        # 写入剩下的27条线的头信息
        if not last:
            exit()
        for trace_no in np.arange((line_num - 27) * cdp_num, line_num * cdp_num,
                                  read_num):  # 遍历文件中的每一道，trace_no 从 0 开始计数
            print('%g - %g' % (trace_no, trace_no + read_num))
            trace_head = get_trace_head(from_file, trace_no, read_num)
            print('道数：%g' % len(trace_head))
            add_info(file2, trace_head)
    print('文件：%s 修改完毕' % filepath)

"""
def output_ts_result(sourcefile=''):
    
    #将文件 sourcefile 的目标层段截取出来，并保存
    #:param sourcefile:
    #:return:
    if '.sgy' in sourcefile:
        ts_filepath = sourcefile[:-4] + '_ts.sgy'
    else:
        ts_filepath = sourcefile + '_ts.sgy'
    with open(sourcefile, 'rb') as file1, open(ts_filepath, 'wb') as file2:
    # 打开文件file1， 并截取目标层段将其输出到file2 中
        # 首先将卷头和前两条线保存下来
        volumn_head = file1.read(3600)
        file2.write(volumn_head)
        first_two_line = file1.read(cdp_num*2*(240+1251*4))
        file2.write(first_two_line)
        for trace_no in range(2*cdp_num, (line_num-27)*cdp_num):
            C_range = get_input_len(trace_no,read_num=1)    # 时间表示的顶和底
            trace_head = file1.read(240)
            Cur_trace = []
            for _ in range(sampling_points):
                Cur_trace.append(struct.unpack('!f',file1.read(4))[0])
            trace_ts = Cur_trace[int(C_range[0]/2):int(C_range[1]/2)]
            trace_ts = [0] * int(C_range[0]/2) + trace_ts
            trace_ts = trace_ts + [0] * (sampling_points - len(trace_ts))
            file2.write(trace_head)
            for point in trace_ts:
                file2.write(struct.pack('!f',float(point)))
"""
if __name__ == '__main__':
    add_trace_head(last=False)
    #output_ts_result()
