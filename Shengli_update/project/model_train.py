'''
启动训练:SVM,BiRNN
'''
from models.Shallowmodel.training.svm_train import *
from models.DLmodel.training.point_to_label.model_training_birnn import *
import sys


def model_train(model=''):
    if model=='BiRNN':
        model_evaluation()
    elif model=='SVM':
        svc()


if __name__ == "__main__":
    model_train(model=sys.argv[1])
    #model_train(model='BiRNN')
