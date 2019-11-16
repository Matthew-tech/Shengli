"""
BiRNN实现对横向切面预测
输入：横向切面文件路径
"""
from models.DLmodel.prediction.point_to_label.label_plane_birnn import *
import sys


if __name__ == '__main__':
    predict_plane(plane_file=sys.argv[1])