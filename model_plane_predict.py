'''
开始预测：切面
'''
from models.Shallowmodel.prediction.svm_predict import *
from models.DLmodel.prediction.point_to_label.label_plane_birnn import *
import sys


def model_plane_predict(plane_file='', model=''):
    if model=='BiRNN':
        predict_plane(plane_file=plane_file, name=str(plane_file.split('/')[-1]))
    elif model=='SVM':
        svc_predict_plane(filepath=plane_file)


if __name__ == "__main__":
    model_plane_predict(plane_file=sys.argv[1], model=sys.argv[2])
    #model_plane_predict(plane_file='data/plane_loc/ng32sz_grid_28jun_154436.p701', model='SVM')
