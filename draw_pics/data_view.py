import pickle
import os
from Configure.global_config import *
import matplotlib.pyplot as plt
import numpy as np
import struct
plane_data_dir = "../data/plane_loc"
cube_data_dir = "/usr/Shengli/"

plane_name = "petrel_Time_gain_attr.sgy_ng33sz_grid_28jun_154331.p701.pkl"
plane_path = os.path.join(plane_data_dir, plane_name)
cube_name = "{}/{}".format("otherseis","petrel_Time_gain_attr.sgy")
cube_name = "{}/{}".format("seismic", "CDD_bigdata_from_petrel.sgy")
cube_name = "{}/{}".format("structureseismic", "petrel_Dip_deviation_attr.sgy")

def slice_up(mode=1):
    """
    :param mode: 1, 表示直接切面，2 表示取上下7，然后求平均
    :return:
    """
    cube_path = os.path.join(cube_data_dir, cube_name)
    with open(plane_path,'rb') as plane_file, open(cube_path,"rb") as cube_file:
        loc_data = pickle.load(plane_file)
        cube_file.read(3600)
        amp_data = []
        for i_inline in range(line_s,line_e+1):
            print(i_inline,line_e+1)
            inline_data = []
            for i_cdp in range(cdp_s, cdp_e+1):
                cube_file.read(240)
                key = "{}-{}".format(i_inline,i_cdp)
                if key in loc_data:
                    if mode == 1:
                        depth = int(float(loc_data[key])/2)
                        cube_file.read((depth-1)*4)
                        inline_data.append(struct.unpack('!f', cube_file.read(4))[0])
                        cube_file.read((sampling_points-depth)*4)
                    elif mode == 2:
                        depth = int(float(loc_data[key])/2) - 7
                        cube_file.read(depth*4)
                        temp = []
                        for _ in range(14):
                            temp.append(struct.unpack('!f', cube_file.read(4))[0])
                        inline_data.append(sum(temp)/14)
                        cube_file.read((sampling_points-depth-14)*4)
                else:
                    cube_file.read(sampling_points*4)
                    inline_data.append(0)

            amp_data.append(inline_data)
        data = np.asarray(amp_data).T
        data = (data-np.min(data))*255/(np.max(data)-np.min(data))
        plt.imshow(data[::-1])
        plt.colorbar()
        title = "cube:{} level:{}".format(cube_name,plane_name.split('.')[1])
        plt.title(title)
        plt.show()
def main():
    slice_up(mode=2)

if __name__ == '__main__':
    main()