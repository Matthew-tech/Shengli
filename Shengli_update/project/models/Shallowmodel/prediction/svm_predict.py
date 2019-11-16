import os
from time import ctime, sleep
import threading
from sklearn.externals import joblib
import pickle as pickle
import numpy as np
import struct
import csv
import matplotlib.pyplot as plt
from Configure.global_config import file_loc_gl
filepath = os.getcwd()

""" File Config """


class FileConfig(object):
    # 文件路径信息

    filepath_seismic = file_loc_gl.seismic_sgy_file_path_base


""" Data Config """


class DataConfig(object):
    # x对应line,y对应trace
    x0, y0 = (640650, 4232695)  # 4232700
    well_nb = 113
    time_nb = 1251
    trace_nb = 664
    line_nb = 1641
    point_nb = 18898  # 23480

    header_nb = 3600
    data_header_byte = 240
    data_content_byte = 4


""" generate main filepath """
mainpath = "/home/eric/workspace/Python_workspace/Shengli_update/project/"#filepath


def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath


""" 高相关性文件路径 """
corr_filepath = os.path.join(mainpath, "data/full_train_data")
""" 模型路径 """
model_path = os.path.join(mainpath, "models/models_weight/SVM/SVC.model")
""" 统计信息路径 """
max_min_mean_std_file = os.path.join(mainpath, "Results/max_min_mean_std_new.pkl")
""" 预测结果保存路径 """
predict_result_filepath = check_folder(os.path.join(mainpath, "Results/point_to_label/SVM/predict_result"))

""" 用于多线程预测的全局变量 """
thread_nb = 30
lines_label = np.ones((thread_nb, DataConfig.trace_nb * DataConfig.time_nb), dtype='float32')


def read_seismic_data_point(filename, line, trace, time):
    """
    read point seismic data
    """
    with open(filename, 'rb') as file:
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            if x == line:
                for y in range(DataConfig.trace_nb):
                    if y == trace:
                        file.read(DataConfig.data_header_byte)
                        temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                        seismic_data = struct.unpack('!' + str(DataConfig.time_nb) + 'f', temp)
                        return seismic_data[time]
                    else:
                        file.read(DataConfig.data_header_byte)
                        file.read(DataConfig.time_nb * DataConfig.data_content_byte)
            elif x < line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)
            else:
                print('read_all_seismic_data Error')
                break

def read_seismic_plane( file,cut_data,times):
    """
    
    """
    plane_data = np.empty((1641, 664), dtype = 'float32')
    with open(file,'rb') as file:
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            for y in range(DataConfig.trace_nb):
                if [x,y] in cut_data:
                    file.read(DataConfig.data_header_byte)
                    temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                    seismic_data = struct.unpack('!'+str(DataConfig.time_nb)+'f',temp)
                    plane_data[x,y] = seismic_data[times[(x,y)]]
                else:
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)

    return plane_data

def read_seismic_plane_data(feature_nb, corr, mean_std, plane_data, times):
    """
    read seismic point data.
    """
    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index

    predict_data = np.empty((1641, 664,feature_nb), dtype = 'float32')

    os.chdir(FileConfig.filepath_seismic)

    child_dir = []
    for dir_name in os.listdir(FileConfig.filepath_seismic):
        if os.path.splitext(dir_name)[1] == '':
            child_dir.append(dir_name)
    index = 0
    for file_dir in child_dir:
        os.chdir(os.path.join(FileConfig.filepath_seismic, file_dir))
        for file in os.listdir(os.path.join(FileConfig.filepath_seismic, file_dir)):
            filename = file.split('.')[0]
            if filename in corr:
                print("index: {}, reading seismic data : {}".format(index,ctime()))
                i = fileindex[file]
                mean = mean_std[file][2]
                std = mean_std[file][3]
                predict_data[:,:,i] = read_seismic_plane( file,plane_data,times)
                predict_data[:,:,i] = (predict_data[:,:,i] - mean) /std
                index += 1

            if index >= feature_nb:
                break
    assert index == feature_nb
    return predict_data

def read_line_seismic_data(filename, line):
    """
    read line seismic data
    """
    seismic_data = np.empty((DataConfig.trace_nb, DataConfig.time_nb), dtype='int32')
    with open(filename, 'rb') as file:
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            if x == line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                    seismic_data[y, :] = struct.unpack('!' + str(DataConfig.time_nb) + 'f', temp)
                return seismic_data
            elif x < line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)
            else:
                print('read_all_seismic_data Error')
                break


def read_corr():
    """
    read correlation file.
    """
    feature_nb = 76
    os.chdir(corr_filepath)
    file = 'high_correlation_attrs.pkl'
    with open(file, 'rb') as f:
        temp = pickle.load(f)

    temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
    corr = []
    for i in range(feature_nb):
        corr.append(temp[i][0].split('.')[0])
    return corr


def model_multithread_predict(index, model, data):
    """
    svc predict.
    """
    global lines_label
    print('predict line:{}, time: {}'.format(index, ctime()))
    temp = model.predict_proba(data[:, :])
    lines_label[index, :] = temp[:, 1]

def read_range_seismic_data(filename, line,line_range=1):
    seismic_data = np.empty((line_range, DataConfig.trace_nb, DataConfig.time_nb), dtype='int32')
    with open(filename,'rb') as file:
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            if x>=line and x< line_range+line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                    seismic_data[x-line,y,:] = struct.unpack('!'+str(DataConfig.time_nb)+'f',temp)
                if x == (line_range + line-1):
                    return seismic_data
            elif x < line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)
            else:
                print('read_all_seismic_data Error')
                break

def read_seismic_point(feature_nb, corr, mean_std, line, trace, time):
    """
    read seismic point data.
    """
    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index

    predict_data = np.empty((feature_nb), dtype=np.float32)
    print("reading seismic data : {}".format(ctime()))
    os.chdir(FileConfig.filepath_seismic)

    child_dir = []
    for dir_name in os.listdir(FileConfig.filepath_seismic):
        if os.path.splitext(dir_name)[1] == '':
            child_dir.append(dir_name)
    index = 0
    for file_dir in child_dir:
        os.chdir(os.path.join(FileConfig.filepath_seismic, file_dir))
        for file in os.listdir(os.path.join(FileConfig.filepath_seismic, file_dir)):

            filename =  file.split('.')[0]
            if filename in corr:
                i = fileindex[file]
                mean = mean_std[file][2]
                std = mean_std[file][3]
                predict_data[i] = read_seismic_data_point(file, line, trace, time)
                predict_data[i] = (predict_data[i] - mean) / std
                index += 1
            if index >= feature_nb:
                break
    assert index == feature_nb

    return predict_data


def read_plane_data(filepath):
    """

    """

    data = []
    times={}
    line_start = 627
    trace_start = 1189
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            datas = row[0].split()
            data.append([int(float(datas[4])-line_start), int(float(datas[3])-trace_start)])
            times[( int(float(datas[4])-line_start), int(float(datas[3])-trace_start) )] = int(float(datas[2])/2)
    return data, times


def svc_predict_point(line, trace, time):
    """
    svc model predict.
    params (line, trace, time) : data point
    """
    predict_prob = np.zeros((1), dtype=np.float32)
    """ load mmodel """
    model = joblib.load(model_path)

    """ read corr file """
    corr = read_corr()

    """ beign predict """
    feature_nb = 76

    predict_data = np.empty((1,feature_nb), dtype=np.float32)

    """  读取index """
    with open(max_min_mean_std_file, 'rb') as file:
        mean_std = pickle.load(file)
    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index
    # 读取地震属性数据点
    predict_data[0,:] = read_seismic_point(feature_nb, corr, mean_std, line, trace, time)
    print(predict_data)
    #     child_dir = []
    #     for dir_name in os.listdir(FileConfig.filepath_seismic):
    #         if os.path.splitext(dir_name)[1] == '':
    #             child_dir.append(dir_name)
    #     index = 0
    #     for file_dir in child_dir:
    #         os.chdir(os.path.join(FileConfig.filepath_seismic, file_dir))
    #         for file in os.listdir(os.path.join(FileConfig.filepath_seismic, file_dir)):
    #             filename = file.split('.')[0]
    #             if filename in corr:
    #                 i = fileindex[file]
    #                 mean = mean_std[file][2]
    #                 std = mean_std[file][3]
    #                 predict_data[i] = read_seismic_data_point( file, line, trace, time)
    #                 predict_data[i] = (predict_data[i] - mean) / std
    #                 index += 1
    #             if index >= feature_nb:
    #                 break
    # assert index == feature_nb

    predict_prob = model.predict_proba(predict_data)
    os.chdir(predict_result_filepath)
    np.save("point_result.npy", predict_prob)
    with open('../../../predict_point.txt', 'w') as file:
        file.write(str(predict_prob[1]))


def svc_predict_plane(filepath="/home/eric/workspace/Python_workspace/Shengli_update/project//data/plane_loc/ng32sz_grid_28jun_154436.p701"):
    """
    svc model predict.
    params (line, trace, time) : data point
    """
    predict_prob = np.zeros((1641, 664), dtype=np.float32)
    """ read cut file """
    cut_data, times = read_plane_data(filepath)

    """ load mmodel """
    model = joblib.load(model_path)

    """ read corr file """
    corr = read_corr()

    """ beign predict """
    feature_nb = 76

    data = np.zeros((1641, 664,feature_nb),dtype=np.float32)

    """  读取index """
    with open(max_min_mean_std_file, 'rb') as file:
        mean_std = pickle.load(file)

    print("reading seismic data...")
    data[:,:,:] = read_seismic_plane_data(feature_nb, corr, mean_std, cut_data, times)
    data = np.reshape(data,(1641*664, feature_nb))



    # 读取地震属性数据
    filename = filepath.split('/')[-1]

    print("predicting ...")
    predict_prob = model.predict_proba(data)

    predict_prob = np.reshape(predict_prob,(1641,664))
    os.chdir(predict_result_filepath)
    np.save("plane_result.npy", predict_prob)
    plt.imshow(predict_prob)
    plt.savefig(filename+"_predict_plane.png")

def svc_predict_all():
    """
    svc model predict.
    params start : start_line for predict
    params line_range : the range of the line for predict
    """
    start = 0
    line_range = 1641

    """ load mmodel """
    print('加载模型')
    model = joblib.load(model_path)

    """ read corr file """
    corr = read_corr()
    print(corr)
    """ beign predict """
    start_time = ctime()
    print('Begin_{}'.format(start))
    feature_nb = 76

    trace_data = np.empty((DataConfig.trace_nb, DataConfig.time_nb), dtype='int32')
    trace_data_multi = np.ones((thread_nb, DataConfig.trace_nb, DataConfig.time_nb), dtype='int32')
    predict_data = np.empty((DataConfig.trace_nb, DataConfig.time_nb, feature_nb), dtype='int32')
    new_time_nb = DataConfig.time_nb

    label = np.empty((line_range, DataConfig.trace_nb, new_time_nb), dtype='float32')
    temp_data = np.zeros((DataConfig.trace_nb * new_time_nb, feature_nb), dtype='float32')
    temp_data_multi = np.ones((thread_nb*DataConfig.trace_nb*new_time_nb,feature_nb), dtype='float32')
    temp = np.zeros((DataConfig.trace_nb * new_time_nb), dtype='float32')
    lines_data = np.zeros((thread_nb, DataConfig.trace_nb * new_time_nb, feature_nb), dtype='float32')

    """  读取index """
    with open(max_min_mean_std_file, 'rb') as file:
        mean_std = pickle.load(file)
    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index

    # 读取地震属性数据，trace
    thread_range = line_range // thread_nb
    rest_line = line_range % thread_nb

    for line in range(start, line_range-rest_line, thread_nb):  # DataConfig.line_nb
        print("reading seismic data : {}".format(ctime()))
        
        os.chdir(FileConfig.filepath_seismic)

        child_dir = []
        for dir_name in os.listdir(FileConfig.filepath_seismic):
            if os.path.splitext(dir_name)[1] == '':
                child_dir.append(dir_name)
        index = 0
        for file_dir in child_dir:
            os.chdir(os.path.join(FileConfig.filepath_seismic, file_dir))
            for file in os.listdir(os.path.join(FileConfig.filepath_seismic, file_dir)):
                filename = file.split('.')[0]
                if filename in corr:
                    trace_data_multi[:, :, :] = read_range_seismic_data(file, line,thread_nb)
                    i = fileindex[file]
                    mean = mean_std[file][2]
                    std = mean_std[file][3]
                    print("index:{}, i:{}".format(index,i))
                    temp_data_multi[:,i] = np.reshape(trace_data_multi,(thread_nb*DataConfig.trace_nb*new_time_nb))
                    temp_data_multi[:, i] = (temp_data_multi[:, i] - mean) / std
                    index += 1
                if index >= feature_nb:
                    break
        assert index == feature_nb

            #------------------
        lines_data[:,:,:] = np.reshape(temp_data_multi,(thread_nb,DataConfig.trace_nb*new_time_nb,feature_nb))

        """ multi threads predicting """
        print("multi threads predicting: {}".format(ctime()))
        threads = []
        for i in range(thread_nb):
            t = threading.Thread(target=model_multithread_predict, args=(i, model, lines_data[i, :, :]))
            threads.append(t)
        for t in threads:
            t.setDaemon(True)
            t.start()

        for t in threads:
            t.join()  # 会阻塞主线程,等待子线程全部跑完再继续下一步，不然值就不会写入，因为主线程不等待。

        print("Round time:{}".format(ctime()))
        for i in range(thread_nb):
            label[line - start + i, :, :] = np.reshape(lines_label[i, :], (DataConfig.trace_nb, new_time_nb))


    """ predict rest data """
    print("reading rest seismic data...")

    """ reading data """
    for line in range(0, rest_line, 1):
        os.chdir(FileConfig.filepath_seismic)

        child_dir = []
        for dir_name in os.listdir(FileConfig.filepath_seismic):
            if os.path.splitext(dir_name)[1] == '':
                child_dir.append(dir_name)
        index = 0
        for file_dir in child_dir:
            os.chdir(os.path.join(FileConfig.filepath_seismic, file_dir))
            for file in os.listdir(os.path.join(FileConfig.filepath_seismic, file_dir)):
                filename = file.split('.')[0]
                if filename in corr:
                    trace_data[:, :] = read_line_seismic_data(file, line_range + line - rest_line)
                    i = fileindex[file]
                    mean = mean_std[file][2]
                    std = mean_std[file][3]
                    # print("index:{}, i:{}".format(index,i))
                    temp_data[:, i] = np.reshape(trace_data, (DataConfig.trace_nb * new_time_nb))

                    temp_data[:, i] = (temp_data[:, i] - mean) / std

                    index += 1
                    if index >= feature_nb:
                        break
        assert index == feature_nb
        print(line)
        lines_data[line, :, :] = temp_data
        
    print("rest multi threads predicting: {}".format(ctime()))

    threads = []
    for i in range(rest_line):
        t = threading.Thread(target=model_multithread_predict, args=(i, model, lines_data[i, :, :]))
        threads.append(t)
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()  # 会阻塞主线程,等待子线程全部跑完再继续下一步，不然值就不会写入，因为主线程不等待。

    for i in range(rest_line):
        label[line_range - rest_line + i, :, :] = np.reshape(lines_label[i, :], (DataConfig.trace_nb, new_time_nb))


    os.chdir(predict_result_filepath)

    header_byte = struct.pack('i', 1)
    with open('svc_prediction.sgy', 'wb') as file:
        for x in range(line_range):  # DataConfig.line_nb
            for y in range(DataConfig.trace_nb):
                for i in range(DataConfig.data_header_byte // 4):
                    file.write(header_byte)
                for z in range(DataConfig.time_nb):
                    byte = struct.pack('!f', label[x, y, z])
                    file.write(byte)
    end_time = ctime()
    print('start_time:{} and end_time:{}'.format(start_time, end_time))


if __name__ == "__main__":
    svc_predict_all()






