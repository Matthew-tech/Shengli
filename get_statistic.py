"""生成归一化文件"""
from preprocess.get_mmms import *
import sys


def get_statistic():
    file_list = get_files()
    get_min_max_mean_std(file_list)


if __name__ == "__main__":
    get_statistic()