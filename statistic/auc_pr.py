#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-10-16 下午1:27
# @Author  : zejin
# @File    : auc_pr.py
import pandas as pd
import matplotlib.pyplot as plt

def auc(FPR, Recall):
    auc_ = 0
    for i in range(101 - 1):
        auc_ += ((Recall[i] + Recall[i + 1]) * abs(FPR[i + 1] - FPR[i])) / 2
    return auc_
model_type = 'birnn'
paras_all = [{'model': model_type, 'trace_range': 0, 'normalize': 'MN',
              'opt': 'Adam', 'loss': 'MSE', 'layers': 1, 'count': 1},
             {'model': model_type, 'trace_range': 0, 'normalize': 'MN',
              'opt': 'Adam', 'loss': 'CEE', 'layers': 1, 'count': 2},
             {'model': model_type, 'trace_range': 1, 'normalize': 'MN',
              'opt': 'Adam', 'loss': 'CEE', 'layers': 1, 'count': 3},
             {'model': model_type, 'trace_range': 1, 'normalize': 'GN',
              'opt': 'Adam', 'loss': 'MSE', 'layers': 1, 'count': 4},
             {'model': model_type, 'trace_range': 0, 'normalize': 'GN',
              'opt': 'Adam', 'loss': 'CEE', 'layers': 3, 'count': 5},
             {'model': model_type, 'trace_range': 0, 'normalize': 'GN',
              'opt': 'Adam', 'loss': 'MSE', 'layers': 2, 'count': 6},
             ]
skip = [17,23,13,1,1,1]
for count in range(1,7):
    sourcefile = '../data_prepare/point_to_label/results_'+str(count)+'.csv'
    data = pd.read_csv(sourcefile)
    true_labels = data['true_label']
    predict_labels = data['predict_label']
    fpr = 0         ; FPR_list = []
    recall = 0      ; Recall_list = []
    precision = 0   ; Precision_list = []
    F1 = 0          ; F1_list = []
    for thres in range(0, 101):
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        thres = thres * 0.01
        for index in range(len(true_labels)):
            tlabel = true_labels.iloc[index]
            plabel = predict_labels.iloc[index]
            if tlabel == 1 and plabel >= thres:
                TP += 1
            elif tlabel == 1 and plabel < thres:
                FN += 1
            elif tlabel == 0 and plabel >= thres:
                FP += 1
            else:
                TN += 1
        if TP + FP == 0:
            Precision = 0
        else:
            Precision = TP / (TP + FP)
        if TP + FN == 0:
            Recall = 0
        else:
            Recall = TP / (TP + FN)
        if Precision + Recall == 0:
            F1 = 0
        else:
            F1 = 2 * Precision * Recall / (Precision + Recall)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)
        Precision_list.append(Precision)
        Recall_list.append(Recall)
        F1_list.append(F1)
        FPR_list.append(FPR)

    Cp = paras_all[count-1]
    """
    plt.plot(FPR_list, Recall_list)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_title = 'ROC Curve (trace_range:%g, Loss func:%s, Norm:%s )'%(Cp['trace_range'],Cp['loss'],Cp['normalize'])
    plt.title(roc_title)
    #plt.show()
    plt.savefig('./roc_pr/'+str(count)+roc_title+'.png',format='png',dpi=1000)
    print(count, auc(FPR_list,Recall_list))
    plt.close()
    """
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    #plt.xticks(range(0,1))
    print(Precision_list)
    Cur_p = Precision_list[:]
    Cur_r = Recall_list[:len(Cur_p)]
    #print(Cur_p)
    plt.plot(Cur_r,Cur_p)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    pr_title = 'PR Curve (trace_range:%g, Loss func:%s, Norm:%s )'%(Cp['trace_range'],Cp['loss'],Cp['normalize'])
    plt.title(pr_title)
    #plt.show()
    plt.savefig('./roc_pr/'+str(count)+pr_title+'.png',format='png',dpi=1000)
    plt.close()
    #exit()