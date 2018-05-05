import os
import sys
import struct
import numpy as np
#导入图表库以进行图表绘制
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator, FuncFormatter
sourceFilePath = 'G:\\我的项目\\2-胜利油田\\数据\\2-地震数据\\'
# 原始地震数据
original_file = 'seismic\\AngleStack18-23'
# 振幅类数据
ampseimsic_file = 'ampseimic\\SingalEnvelope'
#流体检测类属性
flidseis_file = 'S-WaveVelocityReflectivity'
# 相位类属性
phaseseis_file = 'phaseseis\\Amplitude-WeightedApparentPolarity.sgy'

sourceFile = sourceFilePath + original_file


seismic_file = open(sourceFile,'rb')
head = seismic_file.read(3600+240)
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 664*4)
ax.set_ylim(0, 1251)
#ax.set_xticks(range(len(labels)))
ax.set_xticklabels([int(i*(664/6)) for i in range(6)])
#xmajorLocator   = MultipleLocator(100) #将x主刻度标签设置为20的倍数
#xminorLocator   = MultipleLocator(20) #将x轴次刻度标签设置为5的倍数
plane_number = 0
def get_incline_data():
    global plane_number
    trace_data = []
    inline_plane = np.empty((1251,664*4))
    for trace_i in range(664):
        for sampling_i in range(1251):
            Cur_trace_point = seismic_file.read(4)
            trace_data.append(struct.unpack('!f', Cur_trace_point)[0])
        #    print(np.asarray([trace_data,trace_data,trace_data,trace_data]).T.shape)
        inline_plane[:,(trace_i)*4:(trace_i+1)*4] = np.asarray([trace_data,trace_data,trace_data,trace_data]).T[::-1]
        #inline_plane[:,(trace_i)*4:(trace_i+1)*4] = np.asarray([trace_data,trace_data,trace_data,trace_data]).T

        # 跳过道头
        seismic_file.read(240)
        trace_data = []
    plane_number +=1
    return inline_plane

plane_data = plt.imshow(get_incline_data(), animated=True)
def updatefig(*args):
    plane_data.set_array(get_incline_data())
    return plane_data,

def generate_gif():
    #while(plane_number<5):
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    #ani.save('./seismic.gif', writer='imagemagick')
    #ani.save('./flidseis/seismic.gif',writer = 'mencoder')
    plt.show()
    seismic_file.close()
if __name__ == '__main__':
    generate_gif()
