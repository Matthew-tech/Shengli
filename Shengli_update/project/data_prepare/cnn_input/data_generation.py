#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-8 上午9:25
# @Author  : Eric
# @File    : get_input_data_cnn.py

import pickle
import matplotlib.pyplot as plt
#from Configure.global_config import line_s,line_e,cdp_s,cdp_e,sampling_points
import struct
import numpy as np
import os
import csv
from Configure.global_config import file_loc_gl
from preprocess.seismic_data.get_training_seismic_data import get_well_cdp_incline
from data_prepare.point_to_label.data_util import Build_Interpolation_function
cdp_s = 1189
cdp_e = 1852
line_s = 627
line_e = 2267
sampling_points = 1251
total_trace = (line_e-line_s+1)*(cdp_e-cdp_s+1)
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
    image = np.empty(shape=(slice_step,slice_step,3))
    step = int(slice_step / 2)
    # get time slice
    for iline in range(line - step, line + step):
        for icdp in range(cdp - step, cdp + step):
            index = (iline - line_s) * (cdp_e - cdp_s + 1) + (icdp - cdp_s)
            if (index >= 0 and index < total_trace):
                ipoint = 3840 + index * (sampling_points * 4 + 240)
                seismic_file.seek(ipoint + time * 4)
                value = struct.unpack('!f', seismic_file.read(4))[0]
                tmpdata.append(value)
            else:
                tmpdata.append(0)
    image[:,:,0] = np.asarray(tmpdata).reshape((slice_step,slice_step))
#        for i in range(0, slice_step):  # add blank line
#            tmpdata.append(0)
    # get line slice
    tmpdata = []
    for itime in range(time - step, time + step):
        for inline in range(line - step, line + step):
            index = (inline - line_s) * (cdp_e - cdp_s + 1) + (cdp - cdp_s)
            if (index >= 0 and index < total_trace):
                ipoint = 3840 + index * (sampling_points * 4 + 240)
                seismic_file.seek(ipoint + itime * 4)
                value = struct.unpack('!f', seismic_file.read(4))[0]
                tmpdata.append(value)
            else:
                tmpdata.append(0)
        for incdp in range(cdp - step, cdp + step):
            index = (line - line_s) * (cdp_e - cdp_s + 1) + (incdp - cdp_s)
            if (index >= 0 and index < total_trace):
                ipoint = 3840 + index * (sampling_points * 4 + 240)
                seismic_file.seek(ipoint + itime * 4)
                value = struct.unpack('!f', seismic_file.read(4))[0]
                tmpdata.append(value)
            else:
                tmpdata.append(0)
    inline_slice = []
    for i in range(0,slice_step*2,2):
        inline_slice += tmpdata[i*slice_step:(i+1)*slice_step]
    image[:,:,1] = np.asarray(inline_slice).reshape((slice_step,slice_step))
    cdp_slice = []
    for i in range(1,slice_step*2,2):
        cdp_slice += tmpdata[i*slice_step:(i+1)*slice_step]
    image[:,:,2] = np.asarray(cdp_slice).reshape((slice_step,slice_step))
    return image
def get_input_data_cnnpkl():
    sourcefile = '../../data/pics_pkl/traingood.pkl'
    with open(sourcefile,'rb') as file:
        gooddata = pickle.load(file)
        #print(gooddata)
        for _  in range(len(gooddata)):
            plt.imshow(gooddata[0][:,:,0])
            plt.colorbar()
            plt.show()
def generate_cnn_inputs(filepath = '',filename='',total=False):
    """
    生成训练用的数据
    :param filepath:
    :param filename:
    :param total: True表示使用所有的采样点文件，False表示只使用目标层段的采样点
    :return: 将文件保存下来，文件是pkl，保存一个list，every element为一个dict， key=(line,cdp,time),value:np.array
    """
    if total:
        well_sample_file = os.path.join('../../', 'data/1-well/samplint_points_total.csv')
    else:
        well_sample_file = os.path.join('../../','data/1-well/samplint_points.csv')
    if not os.path.join(well_sample_file): save_well_points(total=total)
    this_file = os.path.join(filepath,filename)
    assert os.path.exists(this_file),this_file+'不存在'
    training_dataset_pos = {}
    training_dataset_neg = {}
    training_dataset_dir = '../../data/4-training_data/cnn_train'
    if not os.path.exists(training_dataset_dir): os.makedirs(training_dataset_dir)
    training_file_pos = filename + '_pos.td'
    training_file_neg = filename + '_neg.td'
    if total:
        training_file_pos = training_file_pos[:-3]+'_total'+'.td'
        training_file_neg = training_file_neg[:-3]+'_total'+'.td'

    this_savedfile_pos = os.path.join(training_dataset_dir,training_file_pos)
    this_savedfile_neg = os.path.join(training_dataset_dir,training_file_neg)
    if not os.path.exists(this_savedfile_pos):
        with open(this_file,'rb') as file:
            reader = csv.reader(open(well_sample_file,encoding='utf-8'))
            count = 0
            for line in reader:
                if line[0] == 'well_no': continue
                if count%(100) == 0: print('completing %f'%(count/15000))
                Cur_line = int(line[1])
                Cur_cdp = int(line[2])
                Cur_time = int(line[3])
                is_reservoir = int(line[4])
                Cur_image = get_slice_data(line=Cur_line,cdp=Cur_cdp,time=Cur_time,seismic_file=file)
                if is_reservoir == 1:
                    if line[0] in training_dataset_pos:
                        training_dataset_pos[line[0]].append({(Cur_line,Cur_cdp,Cur_time):Cur_image})
                    else:
                        training_dataset_pos[line[0]] = [{(Cur_line,Cur_cdp,Cur_time):Cur_image}]
                else:
                    if line[0] in training_dataset_neg:
                        training_dataset_neg[line[0]].append({(Cur_line,Cur_cdp,Cur_time):Cur_image})
                    else:
                        training_dataset_neg[line[0]] = [{(Cur_line,Cur_cdp,Cur_time):Cur_image}]
                count += 1
        # 保存正样本的文件
        with open(this_savedfile_pos,'wb') as file:
            print('saving positive training samples...(total:%s)'%str(total))
            pickle.dump(training_dataset_pos,file,-1)
        # 保存负样本的文件
        with open(this_savedfile_neg,'wb') as file:
            print('saving negtive training samples...(total:%s)'%str(total))
            pickle.dump(training_dataset_neg,file,-1)
    else: print(this_savedfile_pos+'已存在！')
def get_horizon_slice(filepath='',filename='',sgyfile=''):
    """
    获得filename 所命名的横向切面的测试数据
    :param filename:
    :return:
    """
    this_file = os.path.join(filepath,filename)     # 横向切面的文件
    #print(os.path.exists(this_file))
    assert os.path.exists(this_file),this_file+'不存在'
    assert os.path.exists(sgyfile),sgyfile+'不存在'
    plane_file_pkl = 'petrel_Time_gain_attr.sgy_'+filename + '.pkl'
    plane_file_pkl_save = os.path.join(filepath,plane_file_pkl)
    if not os.path.exists(plane_file_pkl_save):
        print('正在生成：%s'%plane_file_pkl_save)
        with open(this_file, 'r') as file1, open(plane_file_pkl_save, 'wb') as file2:
            line = file1.readline()
            loc_time_dict = {}
            while line and line != []:
                line_split = line[:-1].split(' ')
                line_split = [value for loc, value in enumerate(line_split) if value != '']
                line_num = line_split[4]
                cdp_num = str(int(float(line_split[3])))
                time = line_split[2]
                loc_time_dict[line_num + '-' + cdp_num] = float(time)
                line = file1.readline()
            #print(loc_time_dict)
            pickle.dump(loc_time_dict, file2,-1)
    with open(plane_file_pkl_save,'rb') as file:
        print('loading...%s'%plane_file_pkl_save)
        loc_time_dict = pickle.load(file) # key:str line-cdp, value:float(time)
    horizon_data = []
    horizon_data_dir = '../../data/4-training_data/cnn_test'
    if not os.path.exists(horizon_data_dir): os.makedirs(horizon_data_dir)
    horizon_data_filename = 'petrel_Time_gain_attr.sgy_'+filename+'.ht'
    this_saved_file = os.path.join(horizon_data_dir,horizon_data_filename)
    if not os.path.exists(this_saved_file):
        with open(sgyfile,'rb') as file:
            count = 0
            for key in loc_time_dict:
                Cur_line = int(key.split('-')[0])
                Cur_cdp = int(key.split('-')[1])
                Cur_time = int(loc_time_dict.get(key)/2)
                if count%(1)==0:print('已完成:%g'%(count/total_trace))

                count += 1
                if count > total_trace//2:
                    Cur_data = get_slice_data(line=Cur_line, cdp=Cur_cdp, time=Cur_time, seismic_file=file)
                    horizon_data.append(Cur_data)
            with open(this_saved_file + '_%g'%count, 'wb') as file:
                pickle.dump(horizon_data, file, -1)

    else:
        print(this_saved_file+'已生成！')
def save_well_points(total=False):
    """
    将井上的标记位置保存下来，保存为csv文件，各列分别为：well_name,line,cdp,time,label
    :return:
    """
    if total:
        well_sample_p_savefile = os.path.join('../../', 'data/1-well/samplint_points_total.csv')
    else:
        well_sample_p_savefile = os.path.join('../../','data/1-well/samplint_points.csv')
    if os.path.exists(well_sample_p_savefile):
        print('%s:已生成'%well_sample_p_savefile)
        return
    else:
        print('正在生成:%s'%well_sample_p_savefile)
    well_loc_filepath = os.path.join('../../',file_loc_gl.well_loc_file)   # csv file
    if total:
        reservoir_Info_filepath = os.path.join('../../',file_loc_gl.well_reservoir_Info_all_clean) # csv file
    else:
        reservoir_Info_filepath = os.path.join('../../',file_loc_gl.well_reservoir_Info_clean)
    well_rel_dir = os.path.join('../../',file_loc_gl.depth_time_rel_dir)
    well_loc_dict = {}
    well_loc_reader = csv.reader(open(well_loc_filepath,encoding='utf-8'))
    for line in well_loc_reader:
        if line[0] == 'well_no': continue
        well_cdp, well_line = get_well_cdp_incline(well_x=float(line[1]),well_y=float(line[2]))
        well_loc_dict[line[0].upper()] = (well_line, well_cdp)
    reservoir_Info_reader = csv.reader(open(reservoir_Info_filepath,encoding='utf-8'))
    well_sample_p = []
    count = 0
    for line in reservoir_Info_reader:
        if line is None or line[0] == 'well_no': continue
        well_n = line[0].upper()
        if well_loc_dict.get(well_n) is None: continue
        if count%100 == 0:print('已完成：%.2f'%(count/3500))
        count += 1
        depth_time_rel_filepath = os.path.join(well_rel_dir,line[0].lower()+'_ck.txt')
        if not os.path.exists(depth_time_rel_filepath):
            depth_time_rel_filepath = os.path.join(well_rel_dir,line[0].upper()+'_ck.txt')
        depth_list, time_list, f = Build_Interpolation_function(depth_time_rel_filepath)

        for time_i in range(int(f(float(line[1]))),int(f(float(line[2])))):
            Cur_sample_point = [line[0],well_loc_dict.get(well_n)[0],well_loc_dict.get(well_n)[1],time_i//2,line[3]]
            if Cur_sample_point not in well_sample_p: well_sample_p.append(Cur_sample_point)

    writer = csv.writer(open(well_sample_p_savefile,'w',newline=''))
    writer.writerow(['well_no','line','cdp','time','is_reservoir'])
    for line in well_sample_p:
        writer.writerow(line)
if __name__ == '__main__':
    filepath='../../data/seismic_data/'
    filename='CDD_bigdata_from_petrel.sgy'
    filepath2 = '/usr/Shengli/otherseis/'
    filename2 = 'petrel_Time_gain_attr.sgy'
    total = True

    #for total in [False,True]:
    #    sgyfile = os.path.join(filepath,filename)
    #    save_well_points(total=total)# 将井上的标记样本保存下来, Total = True 表示将所有采样点的数据进行提取，FALSE表示只使用目标层段
    #    generate_cnn_inputs(filepath=filepath,filename=filename,total=total)
    sgyfile = os.path.join(filepath2, filename2)
    get_horizon_slice(filepath='../../data/plane_loc/',filename='ng32sz_grid_28jun_154436.p701',sgyfile=sgyfile)
