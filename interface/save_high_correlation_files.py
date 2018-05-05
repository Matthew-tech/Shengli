'''
地震数据处理
输入：
     data_path:输入相关性文件路径(sys.argv[1])
'''
from data_prepare.select_high_correlation_attrs import *
import sys


if __name__ == "__main__":
    files.correlation_file = sys.argv[1]
    save_high_correlation_files()