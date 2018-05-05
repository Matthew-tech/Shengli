"""
         ***  Fudan univ. ***
        *** 胜利油田储层预测 ***
"""
print(__doc__)
import os
from models.DLmodel.model.point_to_label.Config import files_deep,model_config
from models.DLmodel.training.point_to_label.model_training_birnn import paras_weight_name
import pickle as pkl
from Configure.global_config import *

def print_status(sys_status):
    for key in sys_status.keys():
        print('**********  '+key+'  Information  **********')
        for key_2 in sys_status.get(key):
            space_num = 30 - len(key_2)
            space = [' ' for _ in range(space_num)]
            print('| '+key_2+ ''.join(space)+str(sys_status.get(key).get(key_2)))

file_status = 'FILE STATUS'
model_status = 'MODEL STATUS'
def get_system_status():
    '''
    返回系统的当前状态，包括各个数据是否已经生成, 模型是否已经训练
    :return: dict
    '''
    sys_status = {}

    sys_status[file_status] = {}
    sys_status[model_status] = {}
    sys_status[file_status].update({'Reservoir Info:':os.path.exists(file_loc_gl.well_reservoir_Info_clean) and
                                                      os.path.exists(file_loc_gl.well_reservoir_Info_all_clean)})

    origin_data = 0
    if os.path.exists(file_loc_gl.full_train_data):
        for child_dir in os.listdir(file_loc_gl.full_train_data):       # seismic
            child_dir_path = os.path.join(file_loc_gl.full_train_data, child_dir)
            if not os.path.isdir(child_dir_path):
                continue
            for attr_file in os.listdir(child_dir_path):
                if os.path.isfile(os.path.join(child_dir_path, attr_file)):
                    origin_data += 1
    sys_status[file_status].update({'Origin Data:':origin_data})


    sys_status[file_status].update({'High correlation attrs pkl:':os.path.exists(os.path.join(file_loc_gl.high_correlation_saved_file))})

    train_data = 0
    if os.path.exists(file_loc_gl.training_data_dir):
        for child_dir in os.listdir(file_loc_gl.training_data_dir):       # seismic
            child_dir_path = os.path.join(file_loc_gl.training_data_dir, child_dir)
            if not os.path.isdir(child_dir_path):
                continue
            for attr_file in os.listdir(child_dir_path):
                if os.path.isfile(os.path.join(child_dir_path, attr_file)):
                    train_data += 1
    sys_status[file_status].update({'Training Data:':train_data})


    # model status:
    paras_dir = 'Results/point_to_label/BiRNN/best_paras'
    if os.path.exists(paras_dir):
        with open(os.path.join(paras_dir, 'best_paras.pkl'), 'rb') as f:
           paras = pkl.load(f)
        weight_name = paras_weight_name(paras)
        #weight_name = 'weights_tracerange_0_layer_1_norm_GN_cell_16_dropout_0.3_GRU_ts_False'#test
        weight_dir = 'models/models_weight/BiRNN'
        flag_weight = False
        if os.path.exists(weight_dir):
            for child_dir in os.listdir(weight_dir):       # seismic
                if weight_name in child_dir:
                    flag_weight = True
    else:
        flag_weight = False
    sys_status[model_status].update({'BiRNN:': flag_weight})
    sys_status[model_status].update({'SVM: ': os.path.exists('models/models_weight/SVM/SVC.model')})
    #sys_status[model_status].update({'Model type:':model_config.model_type})
    #sys_status[model_status].update({'Weighs Location:':model_config.model_graph_weights})

    # weight_nums = 0
    # if os.path.exists(files_deep.weights_dir):
    #     for child_dir in os.listdir(files_deep.weights_dir):
    #         if os.path.isdir(os.path.join(files_deep.weights_dir,child_dir)):
    #             weight_nums += 1
    # sys_status[model_status].update({'Weighs num:':weight_nums})

    print_status(sys_status)
    return sys_status


def sys_info():
    sys_status = get_system_status()
    with open('Results/sys_info.txt', 'w') as file:
        for key in sys_status[file_status].keys():
            file.write(str(sys_status[file_status][key]))
            file.write('\n')
        for key in sys_status[model_status].keys():
            file.write(str(sys_status[model_status][key]))
            file.write('\n')
        file.close()


if __name__ == "__main__":
    #sys_status = get_system_status()
    sys_info()