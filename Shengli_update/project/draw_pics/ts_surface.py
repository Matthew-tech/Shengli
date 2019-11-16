from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def drawpic(type='scatter'):
    source_File_1 = '..\\data\\ts_top_1.csv'
    source_File_2 = '..\\data\\ts_top_2.csv'
    sampling = 10
    df_1 = pd.read_csv(source_File_1)
    df_2 = pd.read_csv(source_File_2)

    data = pd.concat([df_1, df_2])
    x_axis = data['x_axis']
    y_axis = data['y_axis']
    top_value = data['top']
    line = data['line_no']
    trace = data['trace_no']

    X_sampling = np.arange(min(trace.values), max(trace.values) + 1, sampling)
    Y_sampling = np.arange(min(line.values), max(line.values) + 1, sampling)
    top_value = top_value.reshape(int(max(trace.values)-min(trace.values)+1),int(max(line.values)-min(line.values)+1))
    print(top_value.shape)
    top_value_sampling_index = []
    top_value_i = 0
    top_value_j = 0
    for i in range(int(max(trace.values)-min(trace.values)+1)):
        for j in range(int(max(line.values)-min(line.values)+1)):
            if i%sampling == 0:
                if j%sampling == 0:
                    top_value_sampling_index.append(top_value[i][j])
    top_value_sampling = [top_value[i] for i in top_value_sampling_index]
    top_value_sampling = np.asarray(top_value_sampling).reshape(len(Y_sampling),len(X_sampling))
    print(X_sampling.shape, Y_sampling.shape)
    print(top_value_sampling.shape)
    XX, YY = np.meshgrid(X_sampling, Y_sampling)
    if type=='surface':
        fig = plt.figure()
        ax = Axes3D(fig)
        top_value = top_value.tolist()
        top_value = [top_value[i] for i in range(0,len(top_value),sampling)]
        top_value = np.asarray(top_value).reshape(len(Y_sampling),len(X_sampling))
        ax.plot_surface(XX, YY, top_value, rstride=1, cstride=1, cmap='rainbow')
    else:
        fig = plt.figure()
        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        ax.scatter(XX.reshape(1, len(top_value)), YY.reshape(1, len(top_value)), top_value, c='g')
    plt.show()


def main():
    #drawpic(type='scatter')
    drawpic(type='surface')
if __name__ == '__main__':
    main()