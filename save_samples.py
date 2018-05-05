'''
地震数据处理
输入：
     data_path:输入目标层段文件路径(sys.argv[1])
'''
from data_prepare.point_to_label.data_util_shallow import *
import sys


if __name__ == "__main__":
    file_loc_gl.Target_segment_per_well = sys.argv[1]
    save_samples()