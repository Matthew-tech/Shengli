'''
选择井坐标文件
输入：loc_path:井坐标文件路径
'''
from preprocess.well_data.merge_specific_reservoirs import merge_reservoirs
from preprocess.well_data.add_supplement_labels import add_supplement_labels
from preprocess.well_data.add_supplement_labels import clean_reservoir_Info
from Configure.global_config import *
import sys


def select_well_loc(loc_path):
    file_loc_gl.well_loc_file = loc_path
    merge_reservoirs()
    add_supplement_labels()
    clean_reservoir_Info()


if __name__ == "__main__":
    select_well_loc(sys.argv[1])
