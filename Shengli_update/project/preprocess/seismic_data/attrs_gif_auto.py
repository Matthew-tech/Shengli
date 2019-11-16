import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from PIL import Image
from images2gif import writeGif
import os
import sys
import struct
import numpy as np
import scipy
#导入图表库以进行图表绘制
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator, FuncFormatter
sourceFilePath = '../../data/3-seismic_data'
# 原始地震数据
original_file = 'seismic/CDD_bigdata_from_petrel.sgy'
# 振幅类数据
ampseimsic_file = 'ampseimic\\SingalEnvelope'
#流体检测类属性
flidseis_file = 'S-WaveVelocityReflectivity'
# 相位类属性
phaseseis_file = 'phaseseis\\Amplitude-WeightedApparentPolarity.sgy'
# 频率类属性
frequencyseis_file = 'frequencyseis\\StandardDeviationOfFrequency'
attr_files = [original_file,ampseimsic_file,phaseseis_file,frequencyseis_file]
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 664*4)
ax.set_ylim(0, 1251)
#ax.set_xticks(range(len(labels)))
ax.set_xticklabels([int(i*(664/6)) for i in range(6)])
#xmajorLocator   = MultipleLocator(100) #将x主刻度标签设置为20的倍数
#xminorLocator   = MultipleLocator(20) #将x轴次刻度标签设置为5的倍数
def get_all_incline_data(sourceFile,plane_number = 20,direct='vertical'):
    """
    :param sourceFile:
    :param plane_number: 生成的平面个数
    :param direct:
    :return:
    """
    seismic_profile_ = np.empty((plane_number,1251,664*4))

    with open(sourceFile,'rb') as seismic_file:
        head = seismic_file.read(3600 + 240)
        for plane_count in range(plane_number):
            trace_data = []
            inline_plane = np.empty((1251,664*4))
            for trace_i in range(664):
                # 获得一条道数据
                for sampling_i in range(1251):
                    Cur_trace_point = seismic_file.read(4)
                    trace_data.append(struct.unpack('!f', Cur_trace_point)[0])
                #    print(np.asarray([trace_data,trace_data,trace_data,trace_data]).T.shape)
                inline_plane[:,(trace_i)*4:(trace_i+1)*4] = np.asarray([trace_data,trace_data,trace_data,trace_data]).T
                #inline_plane[:,(trace_i)*4:(trace_i+1)*4] = np.asarray([trace_data,trace_data,trace_data,trace_data]).T
                trace_data = []
                # 跳过道头
                seismic_file.read(240)
            #seismic_profile_[plane_count,:,:] = inline_plane[::-1]
            seismic_profile_[plane_count, :, :] = inline_plane
    return seismic_profile_
def vertical_gif():

    for child_dir in os.listdir(sourceFilePath):
        for attr_file in os.listdir(os.path.join(sourceFilePath,child_dir)):

            if not os.path.exists(os.path.join('./vertical_gif_line', child_dir)):  # 生成文件夹
                os.makedirs(os.path.join('./vertical_gif_line', child_dir))
            gif_file_name = os.path.join('./vertical_gif_line/',child_dir,attr_file+'.gif')
            if os.path.exists(gif_file_name):
                print('文件：%s 已存在'%(attr_file+'.gif'))
            else:
                sourceFile = os.path.join(sourceFilePath,child_dir,attr_file)
                print(sourceFile)
                seismic_profile_num = 25
                seismic_profile = get_all_incline_data(sourceFile,seismic_profile_num)
                images = []
                for plane_no in range(len(seismic_profile)):
                    im = Image.fromarray(seismic_profile[plane_no])
                    # scipy.misc.imsave('./images/img_'+str(plane_no)+'.png',im)
                    if im.mode!='RGB':
                        im = im.convert('RGB')
                        #im.save('./images/img_'+str(plane_no)+'.jpg')
                        images.append(im)

                #for image in images:
                #    plt.imshow(image,cmap=plt.cm.gray)
                #    plt.show()
                #images = [Image.open(fn) for fn in os.listdir('./images/')]
                writeGif(gif_file_name, images, duration=0.2)
def horizontal_gif():
    for child_dir in os.listdir(sourceFilePath):
        for attr_file in os.listdir(sourceFilePath + child_dir + '\\'):

            if os.path.exists('./horizontal_gif/' + child_dir + '-' + attr_file + '.gif'):
                continue
            else:
                sourceFile = sourceFilePath + child_dir + '\\' + attr_file
                print(sourceFile)
                seismic_profile_num = 25
                seismic_profile = get_all_incline_data(sourceFile, seismic_profile_num,direct='horizontal')
                images = []
                for plane_no in range(len(seismic_profile)):
                    im = Image.fromarray(seismic_profile[plane_no])
                    # scipy.misc.imsave('./images/img_'+str(plane_no)+'.png',im)
                    if im.mode != 'RGB':
                        im = im.convert('RGB')
                        # im.save('./images/img_'+str(plane_no)+'.jpg')
                        images.append(im)

                # for image in images:
                #    plt.imshow(image,cmap=plt.cm.gray)
                #    plt.show()
                # images = [Image.open(fn) for fn in os.listdir('./images/')]
                gif_name = './horizontal_gif/' + child_dir + '-' + attr_file + '.gif'
                writeGif(gif_name, images, duration=0.2)
if __name__ == '__main__':
    vertical_gif()
    #horizontal_gif()