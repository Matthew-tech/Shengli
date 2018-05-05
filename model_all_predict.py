'''
开始预测：全部
'''
from models.Shallowmodel.prediction.svm_predict import *
from models.DLmodel.prediction.point_to_label.label_all_data_birnn import *
import sys


def model_all_predict(model=''):
    if model=='BiRNN':
        predict_all_area()
    elif model=='SVM':
        svc_predict_all()


if __name__ == "__main__":
    model_all_predict(model=sys.argv[1])
    #model_all_predict(model='SVM')
