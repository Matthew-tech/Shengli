# -*- coding:utf-8 -*-
# time:2018/6/12 下午2:03
# author:ZhaoH
import pickle as pkl
import struct
from Configure.global_config import *
import numpy as np


def trun2dto1d(data):
    re = []
    for ele in data:
        for i in range(len(ele)):
            re.append(ele[i])
    return re


def write2sgy(seismic_data, sgy_file, plane_file):
    """
    :param seismic_data: 原始数据路径
    :param sgy_file: 保存sgy结果路径
    :param plane_file: [顶底文件路径, t0文件， t1文件]
    :return: sgy地震体数据
    """
    file_plane_path = plane_file[0]
    plane_t0 = plane_file[1]
    plane_t11 = plane_file[2]
    plane_file_pkl_s = "{}.pkl".format(plane_t0)
    plane_file_pkl_save_s = os.path.join(file_plane_path, plane_file_pkl_s)
    plane_file_pkl_e = "{}.pkl".format(plane_t11)
    plane_file_pkl_save_e = os.path.join(file_plane_path, plane_file_pkl_e)

    zero_byte = struct.pack('i', 0)
    # get loc_time_dict_s
    with open(plane_file_pkl_save_s, 'rb') as file:
        print('loading...%s' % plane_file_pkl_save_s)
        loc_time_dict_s = pkl.load(file)  # key:str line-cdp, value:float(time)
        print(len(loc_time_dict_s))

    # get loc_time_dict_e
    with open(plane_file_pkl_save_e, 'rb') as file:
        print('loading...%s' % plane_file_pkl_save_e)
        loc_time_dict_e = pkl.load(file)  # key:str line-cdp, value:float(time)
        print(len(loc_time_dict_e))

    with open(seismic_data, 'rb') as seismic_file:
        with open(sgy_file, 'wb') as save_file:
            seismic_file.seek(0)
            print('***********writing head*************')
            volumn_head = seismic_file.read(3600)
            save_file.write(volumn_head)  # 写入文件头 
            total_trace = 0
            for line in range(2 * (cdp_e - cdp_s + 1)): # 跳过前两line
                trace_head = seismic_file.read(240)
                save_file.write(trace_head)
                seismic_file.read(sampling_points * 4)
                for point in range(sampling_points):
                    save_file.write(zero_byte)
            total_trace += 2 * (cdp_e - cdp_s + 1)
            print('***********finished head*************')
            for cur_line_s in range(line_s + 2, line_e - 25 - 4 + 1, 10):
                cur_line_file = "result/cnn_target/pred_line_{}to{}.pkl".format(cur_line_s, cur_line_s+9)
                print("writing: line_{}to{}".format(cur_line_s, cur_line_s+9))
                with open(cur_line_file, 'rb') as cur_file:
                    cur_pred = pkl.load(cur_file)
                    cur_pred = trun2dto1d(cur_pred)
                    #print(cur_pred[:100])
                    #exit()
                    print("cur_pred:", len(cur_pred))
                    cur_point = 0
                    for Cur_line in range(cur_line_s, cur_line_s + 10):
                        for Cur_cdp in range(cdp_s, cdp_e + 1): # 1190-1851 === 1189-1852
                            total_trace += 1
                            if Cur_cdp == cdp_s:
                                trace_head = seismic_file.read(240)
                                save_file.write(trace_head)
                                seismic_file.read(sampling_points * 4)
                                for point in range(sampling_points):
                                    save_file.write(zero_byte)
                                continue
                            if Cur_cdp == cdp_e:
                                trace_head = seismic_file.read(240)
                                save_file.write(trace_head)
                                seismic_file.read(sampling_points * 4)
                                for point in range(sampling_points):
                                    save_file.write(zero_byte)
                                continue
                            trace_head = seismic_file.read(240)
                            seismic_file.read(sampling_points * 4)
                            save_file.write(trace_head)
                            key = "{}-{}".format(Cur_line, Cur_cdp)
                            if key in loc_time_dict_s:
                                if key in loc_time_dict_e:
                                    #print(key)
                                    cur_time_t0 = int(loc_time_dict_s.get(key) / 2)
                                    cur_time_t11 = int(loc_time_dict_e.get(key) / 2)
                                    if cur_time_t0 > cur_time_t11:
                                        print("error")
                                    if cur_time_t0 == cur_time_t11:
                                        print("error")
                                    #print(cur_time_t0, cur_time_t11)
                                    for b in range(cur_time_t0 - 1):
                                        save_file.write(zero_byte)
                                    for b in range(cur_time_t11 - cur_time_t0 + 1):
                                        # print(cur_pred[cur_point])
                                        #exit()
                                        #if (cur_pred[cur_point] - 0.167123913) < 0.0000001:
                                         #   if (cur_pred[cur_point] - 0.167123913) >= 0:
                                         #       print(cur_pred[cur_point])
                                        byte = struct.pack('!f', cur_pred[cur_point])
                                        cur_point += 1
                                        save_file.write(byte)
                                        #print(cur_pred[cur_point])
                                    for b in range(1251 - cur_time_t11):
                                        save_file.write(zero_byte)
                                else:
                                    for point in range(sampling_points):
                                        save_file.write(zero_byte)
                                    cur_point += 1
                                    continue
                            else:
                                for point in range(sampling_points):
                                    save_file.write(zero_byte)
                                cur_point += 1
                                continue
                    print("cur_point: ", cur_point)
            print('***********finished target*************')
            for line in range((2267-2238) * (cdp_e - cdp_s + 1)): # 补全最后的line
                trace_head = seismic_file.read(240)
                save_file.write(trace_head)
                seismic_file.read(sampling_points * 4)
                for point in range(sampling_points):
                    save_file.write(zero_byte)
            total_trace += (2267-2238) * (cdp_e - cdp_s + 1)
            print('***********finished sgy*************')
            print("total_trace: ", (line_e - line_s + 1)*(cdp_e - cdp_s + 1))
            print("generate_trace: ", total_trace)


if __name__ == '__main__':
    seismic_data = "/disk2/Shengli/data/seismicdata/otherseis/petrel_Time_gain_attr.sgy"
    sgy_file = "result/cnn_target/cnn_pred_petrel_Time_gain_attr_test3.sgy"
    plane_info = ["data/plane_loc/", "t1int.txt", "tr.txt"]
    print("start...")
    write2sgy(seismic_data, sgy_file, plane_info)