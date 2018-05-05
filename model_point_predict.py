'''
开始预测：点
'''
from models.Shallowmodel.prediction.svm_predict import *
from models.DLmodel.prediction.point_to_label.label_plane_birnn import predict_point
from Configure.global_config import *
import sys


def model_point_predict(cdp, line, time, model=''):
    if model=='BiRNN':
        predict_point(line_num=line, cdp_num=cdp, time=time)
    elif model=='SVM':
        svc_predict_point(line=(line - line_s), trace=(cdp - cdp_s), time=time)


if __name__ == "__main__":
    model_point_predict(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), model=sys.argv[4])
    #model_point_predict(400, 100, 500, model='SVM')
