'''
地震数据处理
输入：
     data_path:输入原始数据路径(sys.argv[1])
'''

from preprocess.seismic_data.get_training_seismic_data import *
import sys


if __name__ == "__main__":
    files.seismic_sgy_file_path_base = sys.argv[1]
    start_extractting()
