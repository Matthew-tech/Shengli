"""
        ***  2017.09.05  ***
        ***  Fudan univ. ***
        *** 胜利油田储层预测 ***
"""
print(__doc__)
import os
from models.DLmodel.model.point_to_label.Config import files_deep,model_config
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
    sys_status[file_status].update({'Reservoir Info:':os.path.exists(file_loc_gl.well_reservoir_Info_clean)})

    origin_data = 0
    if os.path.exists(file_loc_gl.full_train_data):
        for child_dir in os.listdir(file_loc_gl.full_train_data):       # seismic
            child_dir_path = file_loc_gl.full_train_data.join(child_dir)
            if not os.path.isdir(child_dir_path):
                continue
            for attr_file in os.listdir(child_dir_path):
                if os.path.isfile(child_dir_path.join(attr_file)):
                    origin_data += 1
    sys_status[file_status].update({'Origin Data:':origin_data})

    Attrs_num = 0
    if os.path.exists(files_deep.train_data_dir):
        for j in os.listdir(os.path.join(files_deep.train_data_dir)):  # origin
            for k in os.listdir(os.path.join(files_deep.train_data_dir,j)):    # seg_Interval_30
                for m in os.listdir(os.path.join(files_deep.train_data_dir,j,k)):  # seismic:
                    for filename in os.listdir(os.path.join(files_deep.train_data_dir,j,k,m)):
                        Attrs_num += 1
    sys_status[file_status].update({'Attrs Num for build model:':Attrs_num})

    sys_status[file_status].update({'Top Bottom location pkl:':os.path.exists(os.path.join(file_loc_gl.bottom_top_file_dir,'b_t_dict.pkl'))})
    sys_status[file_status].update({'High correlation attrs pkl:':os.path.exists(os.path.join(file_loc_gl.high_correlation_saved_file))})


    # model status:
    sys_status[model_status].update({'Using Attrs num:':selected_attr_num})
    sys_status[model_status].update({'Model type:':model_config.model_type})
    sys_status[model_status].update({'Weighs Location:':model_config.model_graph_weights})

    weight_nums = 0
    if os.path.exists(files_deep.weights_dir):
        for child_dir in os.listdir(files_deep.weights_dir):
            if os.path.isdir(os.path.join(files_deep.weights_dir,child_dir)):
                weight_nums += 1
    sys_status[model_status].update({'Weighs num:':weight_nums})

    print_status(sys_status)
    return sys_status