import os
import time 
import math
import pickle
import struct
import csv
import numpy as np
import matplotlib.pyplot as plt 
import xgboost as xgb
import threading
from multiprocessing import Pool, sharedctypes
import warnings
import gc
import sys
sys.path.append('/disk3/zk/aboutoil/Shengli/Shengli_update/project')  

from Configure.global_config import file_loc_gl

filepath = os.getcwd()
mainpath = filepath

def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

class FileConfig:
    """
    文件路径管理
    """

    #地震数据文件路径
    filepath_seismic = file_loc_gl.seismic_sgy_file_path_base
    #高相关性文件及路径
    corr_filepath = check_folder(os.path.join(mainpath, "data/full_train_data"))
    corr_file = "high_correlation_attrs.pkl"
    #模型路径
    model_path = os.path.join(mainpath, "models/models_weight/binary_xgboost/binary_xgboost.model")
    #统计信息路径
    max_min_mean_std_file = os.path.join(mainpath, "Results/max_min_mean_std_new.pkl")
    #预测结果保存路径
    predict_result_filepath = check_folder(os.path.join(mainpath, "Results/point_to_label/binary_xgboost/predict_result"))
    #数据保存路径
    plane_savepath = os.path.join(mainpath, "data/plane_loc")

    #平面数据坐标
    # plane_filepath = "/disk2/Shengli_update/project/data/plane_loc/ng32sz_grid_28jun_154436.p701"
    plane_filepath = "/disk3/zk/aboutoil/Shengli/data/plane_loc/ng33sz_grid_28jun_154436.p701"

class DataConfig:
    """
    数据配置管理
    """
    feature_nb = 76
    line_nb = 1641
    time_nb = 1251
    trace_nb = 664

    line_start = 627
    trace_start = 1189

    data_header_byte = 240
    header_nb = 3600
    data_content_byte=4

    thread_nb = 32

#全局变量
multiprocess = True
#global_all_data = np.zeros((DataConfig.line_nb, DataConfig.trace_nb, DataConfig.time_nb, DataConfig.feature_nb), dtype=np.float32)

def lines_predict(line, line_range, Xgboost):
    """
    predict [line, line+line_range] seismic data
    line: start index, >=0
    line_range: read length of line, <=1641
    """
    lines_block = 20
    corr = read_corr()
    print(corr)

    start_time = time.ctime()
    print("Begin_{}".format(line))

    with open(FileConfig.max_min_mean_std_file, 'rb') as file:
        mean_std = pickle.load(file)
    
    means = np.empty((DataConfig.feature_nb), dtype=np.float32)
    stds = np.empty((DataConfig.feature_nb), dtype=np.float32)
    files = {}


    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index


    os.chdir(FileConfig.filepath_seismic)

    child_dir = []
    for dir_name in os.listdir(FileConfig.filepath_seismic):
        if os.path.splitext(dir_name)[1] == '':
            child_dir.append(dir_name)
    index = 0
    for file_dir in child_dir:
        dir_path = os.path.join(FileConfig.filepath_seismic, file_dir)
        for file in os.listdir(dir_path):
            filename = file.split('.')[0]
            if filename in corr:
                i = fileindex[file]
                files[i] = os.path.join(dir_path, file)
                means[i] = mean_std[file][2]
                stds[i] = mean_std[file][3]
                
                index += 1
            if index >= DataConfig.feature_nb:
                break

    assert index == DataConfig.feature_nb


    #trace_data = np.empty((DataConfig.trace_nb, DataConfig.time_nb), dtype=np.int32)
    pred = np.empty((DataConfig.line_nb, DataConfig.trace_nb, DataConfig.time_nb), dtype=np.float32)

    if multiprocess:
        def _init(arr_to_populate):
            global global_plane_data_process
            global_plane_data_process = arr_to_populate
        tmp = np.ctypeslib.as_ctypes(np.zeros((lines_block, DataConfig.trace_nb, DataConfig.time_nb, DataConfig.feature_nb), dtype=np.float32))
        shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)
        p= Pool(processes=32, initializer=_init, initargs=(shared_arr, ))
        for times in range(line_range // lines_block):
        #多进程读取多文件 
            index_start = line + times*lines_block
            print("reading line:{}".format(index_start))
            

            iters = DataConfig.feature_nb
            thread_iters = zip(range(iters), [files[i] for i in range(iters)], \
                                [means[i] for i in range(iters)], [stds[i] for i in range(iters)], \
                                [index_start for _ in range(iters)], [lines_block for _ in range(iters)])
            
            p.map(multiprocess_read_lines, iterable=thread_iters)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pred_data = np.ctypeslib.as_array(shared_arr)
            #np.save(os.path.join(FileConfig.predict_result_filepath,"pred_data.npy"), pred_data)
            index_start = line + times * lines_block
            pred[index_start: index_start+lines_block, :, :] = Xgboost.lines_predict(pred_data, lines_block)
            
            #内存管理
            del pred_data
            gc.collect()
        
        # Rest lines for predict
        res_lines = line_range % lines_block
        if res_lines != 0:
            index_start = line + line_range - res_lines
            print("reading line:{}".format(index_start))

            #def _init(arr_to_populate):
            #    global global_plane_data_process
            #    global_plane_data_process = arr_to_populate

            iters = DataConfig.feature_nb
            thread_iters = zip(range(iters), [files[i] for i in range(iters)], \
                                    [means[i] for i in range(iters)], [stds[i] for i in range(iters)],\
                                    [index_start for _ in range(iters)], [res_lines for _ in range(iters)])
            #tmp = np.ctypeslib.as_ctypes(np.zeros((res_lines, DataConfig.trace_nb, DataConfig.time_nb, DataConfig.feature_nb), dtype=np.float32))
            #shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)
            #p= Pool(processes=32, initializer=_init, initargs=(shared_arr, ))
            p.map(multiprocess_read_lines, iterable=thread_iters)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pred_data = np.ctypeslib.as_array(shared_arr)
        
            pred[index_start: index_start+res_lines, :, :] = Xgboost.lines_predict(pred_data, res_lines)
        end_time = time.ctime()
        print("From {} to {}...".format(start_time,end_time))
        print("Saving .npy file")
        np.save(os.path.join(FileConfig.predict_result_filepath,"pred_lines_{}_{}.npy".format(line,line_range)), pred)
        
        header_byte = struct.pack('i', 1)
        os.chdir(FileConfig.predict_result_filepath)
        with open('svc_prediction_{}_{}.sgy'.format(line, line_range), 'wb') as file:
            for x in range(line_range):  # DataConfig.line_nb
                for y in range(DataConfig.trace_nb):
                    for i in range(DataConfig.data_header_byte // 4):
                        file.write(header_byte)
                    for z in range(DataConfig.time_nb):
                        byte = struct.pack('!f', pred[x, y, z])
                        file.write(byte)
        end_time = time.ctime()
        print('start_time:{} and end_time:{}'.format(start_time, end_time))



def multiprocess_read_lines(args):
    """
    multi threading to read plane data.
    """
    index, file, mean, std, line_start, lines_block = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.ctypeslib.as_array(global_plane_data_process)
    pid = os.getpid()
    print("pid:{} is reading plane data of feature_index:{}".format(pid, index))
    #print("mean:{}, std:{}...".format(mean,std))
    start = time.time()
    plane_feature_data = read_lines_seismic_data(file, line_start, lines_block)
    data[:lines_block,:,:,index] = (plane_feature_data - mean) / std#data[:,:,index]
    end = time.time()
    print("pid:%s runs %0.2f minutes"%(pid, (end - start)/60.0))

def read_lines_seismic_data(filename, line, lines_block = 1):
    """
    read range of lines seismic feature data.
    filename: feature
    line: start index, >=0
    lines_block: read length of line, <=1641
    """
    seismic_data = np.empty((lines_block, DataConfig.trace_nb, DataConfig.time_nb), dtype=np.float32)
    with open(filename, 'rb') as file:
        offset = DataConfig.header_nb + line * DataConfig.trace_nb * (DataConfig.data_header_byte+DataConfig.time_nb * DataConfig.data_content_byte)
        file.seek(offset, 0)
        for i in range(lines_block):
            for y in range(DataConfig.trace_nb):
                file.read(DataConfig.data_header_byte)
                temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                seismic_data[i,y,:] = struct.unpack("!"+str(DataConfig.time_nb)+"f", temp)
        return seismic_data
        """
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            if x >= line and x < lines_block+line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                    seismic_data[x-line,y,:] = struct.unpack("!"+str(DataConfig.time_nb)+"f", temp)
                if x == (lines_block+line-1):
                    return seismic_data
            elif x < line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)
            else:
                print("read_range_seismic_data Error: file/{}".format(filename))
                break
        """


def read_plane_data():
    """
    ensemble read plane function.
    """
    # multithread define global variable 
    if multiprocess == False:
        global  global_plane_data
        global_plane_data = np.zeros((DataConfig.line_nb, DataConfig.trace_nb, DataConfig.feature_nb), dtype=np.float32)

    corr = read_corr()

    cut_loc, times = read_plane_loc(FileConfig.plane_filepath)

    with open(FileConfig.max_min_mean_std_file, 'rb') as file:
        mean_std = pickle.load(file)

    print("reading plane seismic data")
    if multiprocess:
        global_plane_data = read_seismic_plane_data(DataConfig.feature_nb, corr, mean_std, cut_loc, times)
    else:
        read_seismic_plane_data(DataConfig.feature_nb, corr, mean_std, cut_loc, times)
    data = np.reshape(global_plane_data, newshape=(DataConfig.line_nb * DataConfig.trace_nb, DataConfig.feature_nb))

    return data

def read_seismic_plane_data(feature_nb, corr, mean_std, cut_loc, times):
    """
    read seismic plane data of all feature.
    """

    #用于多线程的变量创建
    means = np.empty((DataConfig.feature_nb), dtype=np.float32)
    stds = np.empty((DataConfig.feature_nb), dtype=np.float32)
    files = {}


    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index


    os.chdir(FileConfig.filepath_seismic)

    child_dir = []
    for dir_name in os.listdir(FileConfig.filepath_seismic):
        if os.path.splitext(dir_name)[1] == '':
            child_dir.append(dir_name)
    index = 0
    for file_dir in child_dir:
        dir_path = os.path.join(FileConfig.filepath_seismic, file_dir)
        for file in os.listdir(dir_path):
            filename = file.split('.')[0]
            if filename in corr:
                i = fileindex[file]
                files[i] = os.path.join(dir_path, file)
                means[i] = mean_std[file][2]
                stds[i] = mean_std[file][3]
                
                index += 1
            if index >= feature_nb:
                break

    assert index == feature_nb
    
     
    #多进程读取多文件 
    if multiprocess:
        def _init(arr_to_populate):
            global global_plane_data_process
            global_plane_data_process = arr_to_populate

        iters = DataConfig.feature_nb
        thread_iters = zip(range(iters), [cut_loc for _ in range(iters)], [files[i] for i in range(iters)], \
                           [times for _ in range(iters)], [means[i] for i in range(iters)], [stds[i] for i in range(iters)])
        tmp = np.ctypeslib.as_ctypes(np.zeros((DataConfig.line_nb, DataConfig.trace_nb, DataConfig.feature_nb), dtype=np.float32))
        shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)
        p= Pool(processes=32, initializer=_init, initargs=(shared_arr, ))
        p.map(multiprocess_read_plane, iterable=thread_iters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pred_data = np.ctypeslib.as_array(shared_arr)

        return pred_data
       
    
    #多线程读取多文件，python GIL限制导致无法多线程读取
    else:
        left_thread = DataConfig.feature_nb
        thread_nb = DataConfig.thread_nb
        for epoch in range(math.ceil(DataConfig.feature_nb/thread_nb)):
            if left_thread >= thread_nb:
                thread_nb = DataConfig.thread_nb
            else:
                thread_nb = left_thread

            left_thread = left_thread - thread_nb
            threads = []
            for i in range(thread_nb):
                index = i + epoch * DataConfig.thread_nb
                t = threading.Thread(target=multithread_read_plane, args=(index, cut_loc, files[index], times, means[index], stds[index]))
                threads.append(t)
            for t in threads:
                t.setDaemon(True)
                t.start()
            for t in threads:
                t.join()

        assert left_thread == 0
        
    

def multiprocess_read_plane(args):
    """
    multi threading to read plane data.
    """
    index, cut_loc, file, times, mean, std = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.ctypeslib.as_array(global_plane_data_process)
    pid = os.getpid()
    print("pid:{} is reading plane data of feature_index:{}".format(pid, index))
    #print("mean:{}, std:{}...".format(mean,std))
    start = time.time()
    plane_feature_data = read_seismic_plane(file,cut_loc,times)
    data[:,:,index] = (plane_feature_data - mean) / std#data[:,:,index]
    end = time.time()
    print("pid:%s runs %0.2f seconds"%(pid, (end - start)/3600.0))
    
def multithread_read_plane(index, cut_loc, file, times, mean, std):
    """
    multi threading to read plane data.
    """
    print("index:{}, mean:{}, std:{}...".format(index,mean,std))
    start = time.time()
    global_plane_data[:,:,index] = read_seismic_plane(file,cut_loc,times)
    data[:,:,index] = (data[:,:,index] - mean) / std
    end = time.time()
    print("pid:%s runs %0.2f seconds"%(pid, (end - start)/3600.0))
    

def read_seismic_plane(file, cut_loc, times):
    """
    read seismic plane of each feature.
    """

    plane_data = np.empty((DataConfig.line_nb, DataConfig.trace_nb), dtype=np.float32)
    with open(file, 'rb') as file:
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            for y in range(DataConfig.trace_nb):
                if [x,y] in cut_loc:
                    file.read(DataConfig.data_header_byte)
                    """
                    temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                    seismic_data = struct.unpack('!'+str(DataConfig.time_nb) + 'f', temp)
                    plane_data[x,y] = seismic_data[times[(x,y)]]
                    """
                    skip_nb = times[(x,y)] 
                    res_nb = DataConfig.time_nb - skip_nb - 1
                    file.read(skip_nb * DataConfig.data_content_byte)
                    temp = file.read(DataConfig.data_content_byte)
                    seismic_data = struct.unpack('!f', temp)
                    plane_data[x,y] = seismic_data[0] 
                    file.read(res_nb * DataConfig.data_content_byte)
                else:
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)

    return plane_data

def read_plane_loc(file):
    """
    read plane location.
    """
    data = []
    times = {}

    with open(file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            datas = row[0].split()
            data.append([int(float(datas[4]) - DataConfig.line_start), int(float(datas[3]) - DataConfig.trace_start)])
            times[( int(float(datas[4]) - DataConfig.line_start), int(float(datas[3]) - DataConfig.trace_start) )] = int(float(datas[2])/2)
    return data, times

def read_corr():
    """
    read correlation file.
    """
    os.chdir(FileConfig.corr_filepath)
    with open(FileConfig.corr_file, 'rb') as f:
        temp = pickle.load(f)

    temp = sorted(temp.items(), key=lambda x:x[1], reverse=True)
    corr=[]
    for i in range(DataConfig.feature_nb):
        corr.append(temp[i][0].split(".")[0])

    return corr

class Xgboost_Predictor:
    """
    Predictor of Xgboost
    """
    def __init__(self):
        """
        Predictor constructer.
        """
        self.model_path = os.path.join(filepath, "models/models_weight/binary_xgboost/binary_xgboost.model")
        init_params = {
                    "nthread":30
                        }
        if os.path.exists(self.model_path):
            self.model = xgb.Booster(params = init_params, model_file=self.model_path)
        else:
            print("model doesn't exist! please train xgboost model.")

    def plane_predict(self):
        """
        predict plane data.
        """
        data = read_plane_data()
        np.save(os.path.join(FileConfig.plane_savepath,"plane_data.npy"), data)
        print("predict...")
        xg_data = xgb.DMatrix(data)
        pred = self.model.predict(xg_data)
        pred = np.reshape(pred, newshape=(DataConfig.line_nb, DataConfig.trace_nb))

        os.chdir(FileConfig.predict_result_filepath)
        np.save("plane_result.npy", pred)

        filename = FileConfig.plane_filepath.split("/")[-1]
        plt.imshow(pred)
        plt.savefig("{}_predict_plane.png".format(filename))

    def lines_predict(self, line_data, lines_block):
        """
        predict lines data.
        line_data: input
        lines_block: number of lines
        """
        line_data = np.reshape(line_data[0:lines_block], newshape=(lines_block * DataConfig.trace_nb * DataConfig.time_nb, DataConfig.feature_nb))
        xg_data = xgb.DMatrix(line_data)
        pred = self.model.predict(xg_data)
        pred = np.reshape(pred, newshape=(lines_block, DataConfig.trace_nb, DataConfig.time_nb))

        del line_data
        del xg_data
        gc.collect()

        return pred

data = read_plane_data()
np.save(os.path.join(FileConfig.plane_savepath,"/disk3/zk/aboutoil/Shengli/data/plane_loc/plane_data_ng33.npy"), data)