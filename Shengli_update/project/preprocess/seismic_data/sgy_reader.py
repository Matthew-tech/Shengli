import os
import struct
import numpy as np
import pandas as pd

#导入图表库以进行图表绘制
import matplotlib.pyplot as plt


def main():
    # 可以读取制定线号和道号的地震数据
    sourceFilePath = 'G:\\我的项目\\2-胜利油田\\数据\\2-地震数据\\seismic\\'
    sourceFileName = 'AngleStack(30-35)_all'
    sourceFile = sourceFilePath+sourceFileName
    with open(sourceFile, 'rb') as seismic_file:
        # 读取头部信息
        head = seismic_file.read(3600 + 240)
        line_no_input = input('请输入线号：\n')
        trace_no_input = input('请输入道号：\n')
        inline_plane = np.empty((1251, 664))
        # 跳过前面的线面
        for line_no in range(int(line_no_input)):
            for trace_i in range(664):
                for sampling_i in range(1251):
                    seismic_file.read(4)
                seismic_file.read(240)
        # 跳过前面的道
        for trace_i in range(int(trace_no_input)):
            for sampling_i in range(1251):
                seismic_file.read(4)
            seismic_file.read(240)
        # 读当前对应线号和道号的 trace_data
        trace_data = []
        for sampling_i in range(1251):
            Cur_trace_point = seismic_file.read(4)
            trace_data.append(struct.unpack('!f', Cur_trace_point)[0])

        plt.plot(trace_data)
        plt.xlabel('time')
        plt.ylabel('amptitude')
        plt.title(sourceFileName+'       line_no:'+line_no_input+'trace_no:'+trace_no_input)
        plt.show()
if __name__ == '__main__':
    main()