import csv
import os
from DLmodel.model.Config import *
import numpy as np
import matplotlib.pyplot as plt
def duplicate(Cur_line_mse,duplicate_factor):
    Cur_line_mse_re = []
    for value in Cur_line_mse:
        for _ in range(duplicate_factor):
            Cur_line_mse_re.append(value)
    return Cur_line_mse_re
duplicate_factor = 8
def get_mse_array(mse_file):
    mse_array = []

    with open(mse_file, 'r') as file:
        reader = csv.reader(file, dialect='excel')
        for line in reader:
            if line[0] == 'Error_total' or line[0] == 'Trace_no':
                continue
            Cur_line_mse = list(map(float, line[3:]))
            Cur_line_mse.extend([0] * (SEQ_LEN - int(line[2])))
            Cur_line_mse = duplicate(Cur_line_mse, duplicate_factor)
            mse_array.append(Cur_line_mse)
    mse_array = np.asarray(mse_array).T
    return mse_array
def imshow_mse():
    error_file_dir = 'E:\Programming\Python_workspace\Shengli\Results\BiRNN\origin\input_dim_13\Interval_25ms'
    y_list = list(np.asarray(range(0, 21, 2)) * duplicate_factor)
    y_tick = list(map(str, range(0, 21, 2)))
    for reservoir_range in range(3):
        mse_file_name = 'results_ts_range_' + str(reservoir_range) + '_op_average_max_r_20_tf_True_mse_new.csv'
        mse_file = os.path.join(error_file_dir, mse_file_name)
        Cur_mse_array = get_mse_array(mse_file)
        # all_error_array.append(Cur_mse_array)
        plt.subplot(3, 1, reservoir_range + 1)
        plt.imshow(Cur_mse_array, cmap='gray')
        if reservoir_range == 3 - 1:
            plt.xlabel('Trace No.')
        plt.ylabel('Reservoir No.')
        plt.yticks(y_list, y_tick)
        plt.colorbar()
        plt.title('MSE plane(range=%s)' % reservoir_range)
    plt.show()
def get_mse_trace(mse_file):
    mse_trace = []
    reservoir_num = []
    with open(mse_file, 'r') as file:
        reader = csv.reader(file, dialect='excel')
        for line in reader:
            if line[0] == 'Error_total' or line[0] == 'Trace_no' or line[0] == 'mse_total':
                continue
            mse_trace.append(float(line[1]))
            reservoir_num.append(int(line[2]))
    return mse_trace, reservoir_num
def plot_mse_trace():
    # BiRNN
    error_file_dir = 'E:\Programming\Python_workspace\Shengli\Results\BiRNN\origin\input_dim_13\Interval_25ms'
    # Poly
    #error_file_dir = 'E:\Programming\Python_workspace\Shengli\Results\Polynomial_regression\origin\input_dim_13\Interval_25ms'
    font_size = 20
    for reservoir_range in range(1):
        # BiRNN
        mse_file_name = 'results_ts_range_' + str(reservoir_range) + '_op_average_max_r_20_tf_True_mse_new.csv'
        # Poly
#        mse_file_name = 'results_ts_range_'+str(reservoir_range)+'_op_average_max_r_20_pr3_mse.csv'
        mse_file = os.path.join(error_file_dir, mse_file_name)
        Cur_mse_trace,reservoir_num = get_mse_trace(mse_file)
        fig = plt.figure(figsize=(25, 20))
        ax1 = fig.add_subplot(111)
        # 绘制Total曲线图
        ax1.plot(Cur_mse_trace,'b')
#        ax1.scatter(range(len(Cur_mse_trace)),Cur_mse_trace, color='b')
        # 设置双坐标轴，右侧Y轴
        ax2 = ax1.twinx()
#        ax2.plot(reservoir_num,'r')
        ax2.scatter(range(len(Cur_mse_trace)),reservoir_num,color = 'r')
        ax1.set_xlabel('Trace No.', fontsize=font_size)
        ax1.set_ylabel('MSE per trace', fontsize=font_size)
        ax2.set_ylabel('reservoirs num per trace', fontsize=font_size)
        plt.title('mse per trace(range=%s)'%reservoir_range)
        plt.show()
def plot_predict_true_label():
    label_dir = 'E:\Programming\Python_workspace\Shengli\Results\BiRNN\origin\input_dim_13\Interval_25ms'
    for r_range in range(0,3):
        label_filename = 'results_ts_range_'+str(r_range)+'_op_average_max_r_20_tf_True_mse_new_predict_labels_new.csv'
        # 第一行是预测值，第二行是真实值
        predict_label_list = []
        true_label_list = []
        with open(os.path.join(label_dir,label_filename),mode='r') as file:
            reader = csv.reader(file,dialect='excel')
            flag = 0    # flag = 0 表示当前行是预测值，flag = 1 表示当前行是真实值
            for line in reader:
                if line[0] == 'Trace_no':
                    continue
                line = list(map(float,line))
                if flag == 0:
                    line = list(map(lambda x:x/(r_range+1), line))
                if line[-1] == -1:
                    to_be_extend = line[1:-1]
                else:
                    to_be_extend = line[1:]
                if flag == 0:
                    predict_label_list.extend(to_be_extend)
                else:
                    true_label_list.extend(to_be_extend)
                flag = 1 if flag == 0 else 0
        plt.subplot(3,1,1)
        plt.plot(predict_label_list,color='b')
        plt.plot(true_label_list,color='r')
        plt.legend(['predict labels','true labels'])
        plt.ylabel('Reservoir proportion')
        plt.title('Predict labels and true labels comparasion(grid:%s*%s)'%((r_range+1)**2,(r_range+1)**2))

        plt.subplot(3,1,2)
        plt.plot(predict_label_list[:100], color='b')
        plt.plot(true_label_list[:100], color='r')
        plt.ylabel('Reservoir proportion')
        plt.legend(['predict labels', 'true labels'])
        plt.subplot(3,1,3)
        plt.plot(predict_label_list[100:200], color='b')
        plt.plot(true_label_list[100:200], color='r')
        plt.xticks(range(0,100,20),range(100,200,20))
        plt.xlabel('Reservoir No.')
        plt.ylabel('Reservoir proportion')
        plt.legend(['predict labels', 'true labels'])

        plt.show()

def main():
    #imshow_mse()
    #plot_mse_trace()
    plot_predict_true_label()
if __name__ == '__main__':
    main()