import sys

#sys.path.append("/disk2/Shengli/project/")
from data_prepare.point_to_label.data_util_shallow import *
import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os
# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

plt.switch_backend('agg')

filepath = os.getcwd()

""" generate main filepath """
mainpath = filepath
#for i in range(3):
#    mainpath = os.path.split(mainpath)[0]


def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath


""" 测试结果保存路径 """
test_result_filepath = check_folder(os.path.join(mainpath, "Results/point_to_label/SVM/test_result"))
""" 测试结果图保存路径 """
test_presentation_filepath = check_folder(os.path.join(mainpath, "Results/point_to_label/SVM/test_presentation"))
""" 模型保存路径 """
model_path = os.path.join(mainpath, "models/models_weight/SVM/SVC.model")


def svc():
    """
    SVC Model.
    """
    print('svm start training')
    """ Train and choose best params. """
    train_data, validation_data, test_data = get_input(paras = {'norm':'GN','ts':False})
    train_x = np.concatenate([train_data[0], validation_data[0]])
    train_y = np.concatenate([train_data[1], validation_data[1]])
    test_x = test_data[0]
    test_y = test_data[1]
    samples = test_data[2]

    params = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1], 'kernel': ['rbf']}
    gs_all = GridSearchCV(estimator=SVC(probability=True), param_grid=params, scoring='neg_log_loss', cv=10, n_jobs=20)
    gs_all.fit(train_x, np.ravel(train_y))

    """ train best_params' model """
    print(gs_all.grid_scores_)
    print(gs_all.best_params_)
    print(gs_all.best_score_)

    c = gs_all.best_params_["C"]
    g = gs_all.best_params_["gamma"]
    k = gs_all.best_params_["kernel"]
    best_svc = SVC(C=c, gamma=g, kernel=k, probability=True)
    best_svc.fit(train_x, np.ravel(train_y))

    """ save best model """

    joblib.dump(best_svc, model_path)

    """ predict test_data """
    pred = best_svc.predict_proba(test_x)

    """ save predict result to csv """
    os.chdir(test_result_filepath)
    with open('pre_label.csv', 'w') as file:
        write = csv.writer(file)
        for i in range(len(test_x)):
            row = []
            row.append(pred[i][1])
            row.append(test_y[i])
            write.writerow(row)

    """ metric evaluate """

    os.chdir(test_result_filepath)
    with open('evaluate_metrics.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['Threshold', 'TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'FDR', 'TDR'])
        for i in range(200):
            threshold = i / 199
            evaulate(threshold)
            (TP, TN, FP, FN), (precision, recall), (FPR, TPR) = calc_metrics()
            writer.writerow([threshold, TP, TN, FP, FN, precision, recall, FPR, TPR])

    """ plot PR and ROC Curve """
    evaluate_plot()
    
    #presentation(samples)


def convert(data, threshold):
    conv_data = []
    if float(data[0]) >= threshold:
        conv_data.append(1)
        conv_data.append(int(float(data[1])))
    else:
        conv_data.append(0)
        conv_data.append(int(float(data[1])))
    return conv_data


def evaulate(threshold, filename='pre_label.csv'):
    with open(filename) as readfile, open('process_' + filename, 'w') as writefile:
        writer = csv.writer(writefile, lineterminator='\n')
        content = csv.reader(readfile)
        for i, line in enumerate(content):
            data = convert(line, threshold)
            writer.writerow(data)


def calc_metrics(filename='process_pre_label.csv'):
    epsilon = 0.1
    TP, FP, TN, FN = (0, 0, 0, 0)
    Acc = 0
    nb = 0
    with open(filename) as file:
        content = csv.reader(file)
        for data in content:
            nb += 1
            if int(data[0]) == 1:
                if int(data[1]) == 1:
                    Acc += 1
                    TP += 1
                else:
                    FP += 1
            else:
                if int(data[1]) == 0:
                    Acc += 1
                    TN += 1
                else:
                    FN += 1
    precision = (epsilon + TP) / (epsilon + FP + TP)
    recall = (epsilon + TP) / (epsilon + TP + FN)
    FPR = (epsilon + FP) / (epsilon + FP + TN)
    TPR = (epsilon + TP) / (epsilon + TP + FN)
    return (TP, TN, FP, Acc / nb), (precision, recall), (FPR, TPR)


def evaluate_plot():
    num = 1
    Precision = np.empty((num, 200), dtype='float32')
    Recall = np.empty((num, 200), dtype='float32')
    FPR = np.empty((num, 200), dtype='float32')
    TPR = np.empty((num, 200), dtype='float32')
    # plt.figure(1,figsize=(10,10))

    os.chdir(test_result_filepath)
    j = 0
    i = 0
    with open('evaluate_metrics.csv', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if line[0] != 'Threshold':
                Precision[i][j] = float(line[5])
                Recall[i][j] = float(line[6])
                FPR[i][j] = float(line[7])
                TPR[i][j] = float(line[8])
                j += 1

    os.chdir(test_presentation_filepath)

    """ plot ROC/PR Curve """
    # plt.subplot(1,1,1)
    plt.plot(FPR[i, :], TPR[i, :],label='ROC')
    plt.plot(Recall[i, :], Precision[i, :],label='PR')
    plt.title('SVC ROC/PR Curve', fontsize=15)
    plt.xlabel('FPR/Recall', fontsize=15)
    plt.ylabel('TPR/Precesion', fontsize=15)
    legend = plt.legend(loc='center left', bbox_to_anchor=(0.85, 0.3), borderpad=0.1, labelspacing=0.1)
    plt.savefig('SVC_ROC_PR.png', dpi=100)
    plt.close('all')

        
def presentation(samples):
    test_range = []
    for wellname in samples.keys():
        test_range.append(len(samples[wellname][0]))
    filename = 'evaluate_metrics.csv'
    
    SVC_path = test_result_filepath
    save_path = test_presentation_filepath
    pred = []
    label = []

    xc = []
    correct = []
    xw = []
    wrong = []
    x1 = 0
    x2 = -1
    os.chdir(SVC_path)

    with open('pre_label.csv','r') as file:
        content = csv.reader(file)
        for i,line in enumerate(content):
            if float(line[0]) > 0.5:
                pred.append((1))
            else:
                pred.append(0)
            label.append(float(line[1]))

    for i,well_range in enumerate(test_range[:-1]):
        a = np.arange(0,well_range,1)
        x2 = x1
        x1 = x1 + well_range
        width = 100
        plt.figure(2,figsize=(10,15))
        if i == 0:
            pred1 = np.repeat(pred[-1*x1:],width)
            pred2 = np.reshape(pred1, (well_range,width))
            label1 = np.repeat(label[-1*x1:], width)
            label2 = np.reshape(label1, (well_range, width))
            plt.subplot(221)
            plt.imshow(label2)
            plt.subplot(222)
            plt.imshow(pred2)
            
            os.chdir(save_path)
            plt.title('Well {} Result Presentation'.format(i),fontsize=15)
            plt.xlabel("Well Depth")
            plt.ylabel("Reservoir")
            legend = plt.legend(loc='center left',bbox_to_anchor=(0.9,0.9),borderpad=0.1,labelspacing=0.1)
            os.chdir(save_path)
            plt.savefig('Well {}.png'.format(i),dpi=100)
            plt.show()
            plt.close('all')
        else:
            pred1 = np.repeat(pred[-1*x1:-1*x2],width)
            if well_range<(len(pred1)//width):
                well_range = int(len(pred1)//width)
                print(i)
            pred2 = np.reshape(pred1, (well_range,width))
            label1 = np.repeat(label[-1*x1:-1*x2], width)
            label2 = np.reshape(label1, (well_range, width))
            plt.subplot(221)
            plt.imshow(label2)
            plt.subplot(222) 
            plt.imshow(pred2)
            
            os.chdir(save_path)
            plt.title('Well {} Result Presentation'.format(i),fontsize=15)
            plt.xlabel("Well Depth")
            plt.ylabel("Reservoir")
            legend = plt.legend(loc='center left',bbox_to_anchor=(0.9,0.9),borderpad=0.1,labelspacing=0.1)
            os.chdir(save_path)
            plt.savefig('Well {}.png'.format(i),dpi=100)
            plt.show()
            plt.close('all')


if __name__ == '__main__':
    #svc()
    train_data, validation_data, test_data = get_input()
    samples = test_data[2]
    presentation(samples)