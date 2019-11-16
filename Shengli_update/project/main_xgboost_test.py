from models.Shallowmodel.training.xgboost_train import *
from models.Shallowmodel.prediction.xgboost_predict import *
from data_prepare.point_to_label.data_util_shallow import *
import numpy as np

if __name__ == "__main__":
    # xgboost train: params search
    #xgboost_search_params()
    
    # predict 
    predict = Xgboost_Predictor()
    # plane data 
    #predict.plane_predict()
    # lines data
    lines_predict(0, 1641, predict)#1641-850