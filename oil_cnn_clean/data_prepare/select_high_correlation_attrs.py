import os
import csv
import pickle
import shutil
from Configure.global_config import *
files = file_loc_global()
def get_correlation_dict():
    correlation_dict = {}
    with open(files.correlation_file,'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if line[0] == 'Attr':
                continue
            else:
                key = line[0].replace(' ','_')
                correlation_dict.update({key.lower():float(line[1])})
    return correlation_dict
def attr_is_high_correlation(key,filename):

    # key 和 filename是一对一的关系,需要完全匹配
    if 'petrel_' in filename:
        filename = filename.replace('petrel_','')
    if '.sgy' in filename:
        filename = filename.replace('.sgy','')
    if '_attr' in filename:
        filename = filename.replace('_attr','')
    for sub_str in key.split('_'):
        if sub_str not in filename.lower():
            return False
    # 统计key中字符的个数和filename中字符个数的比较
    key_char_num = sum([1 for i in key if i!='_'])
    filename_char_num = sum([1 for i in filename if i!='_'])
    if abs(key_char_num-filename_char_num)<=3:
        return True
    return False

def check_is_high_correlation(child_dir,attr_file):
    #print(child_dir)
    #print(attr_file)
    correlation_dict = get_correlation_dict()
    #print(correlation_dict)
    for key in sorted(correlation_dict.keys()):
        if '.pkl' in attr_file:
            if attr_is_high_correlation(key,attr_file[attr_file.index(child_dir)+len(child_dir)+1:-4]):
                return True,key
        else:
            if attr_is_high_correlation(key, attr_file[attr_file.index(child_dir) + len(child_dir) + 1:]):
                return True,key
    return False,-1
def save_high_correlation_files():
    attrs_dir = files.seismic_sgy_file_path_base
    match_list = {}
    if os.path.exists(files.high_correlation_saved_file):
        print('高相关性文件已存在！！！')
    else:
        for child_dir in os.listdir(attrs_dir):
            if not os.path.isdir(os.path.join(attrs_dir,child_dir)):
                continue
            for attr_file in os.listdir(os.path.join(attrs_dir,child_dir)):
                match, matched_high_correlation = check_is_high_correlation(child_dir,child_dir + '_' + attr_file)
                #print(attr_file)
                if match:
                    if matched_high_correlation not in match_list.keys():
                        filename_list = [os.path.join(attrs_dir,child_dir,attr_file)]
                        filename_list = [attr_file]
                    else:
                        filename_list = match_list.get(matched_high_correlation)
                        #filename_list.append(os.path.join(attrs_dir,child_dir,attr_file))
                        filename_list.append(attr_file)
                    match_list.update({matched_high_correlation:filename_list})
        # 输出所有匹配成功的key 和 filename
        for key in sorted(match_list.keys()):
            print(key,len(match_list.get(key)),'\n')#,match_list.get(key))

        # 输出尚未匹配的key值
        print('\n******************\n    尚未匹配出的key：%g'%(len(get_correlation_dict().keys())-len(match_list.keys())))
        for key in sorted(get_correlation_dict().keys()):
            if key not in match_list.keys():
                print(key)
        # 将match_list 的value作为key，get_correlation_dict()的value作为 value 保存到本地
        high_correlation_attrs_dict = {}
        for key in match_list.keys():
            high_correlation_attrs_dict.update({match_list.get(key)[0]:get_correlation_dict().get(key)})
        with open(files.high_correlation_saved_file,'wb') as saved_file:
            pickle.dump(high_correlation_attrs_dict,saved_file,-1)
def read_high_correlation_files():
    with open(files.high_correlation_saved_file,'rb') as file:
        correlation_data = pickle.load(file)
        return correlation_data

'''     # 将相关性低的文件移动到 low_correlation_dir里面
def move_into_low_dir(*args,filename='',data_type='train'):
    file_path_from = files.train_data_dir
    file_path_to = os.path.join(files.low_correlation_dir,'normalize_'+str(data_config.is_normalize),data_type)
    for sub_para in args:
        file_path_to = os.path.join(file_path_to,sub_para)
        file_path_from = os.path.join(file_path_from, sub_para)
    if not os.path.exists(file_path_to):
        os.makedirs(file_path_to)
    shutil.move(os.path.join(file_path_from,filename),file_path_to)
def select_high_correlation_files(data_dir, correlation_dict, data_type = 'train'):
    for featype_dir in os.listdir(data_dir):
        for seg_Interval_dir in os.listdir(os.path.join(files.train_data_dir,featype_dir)):
            for attr_dir in os.listdir(os.path.join(files.train_data_dir,featype_dir,seg_Interval_dir)):
                for filename in os.listdir(os.path.join(files.train_data_dir,featype_dir,seg_Interval_dir,attr_dir)):
                    print(filename[:filename.index('_ts')])
                    if not filename[:filename.index('_ts')] in correlation_dict.keys():
                        print(filename[:filename.index('_ts')])

                        move_into_low_dir(featype_dir,seg_Interval_dir,attr_dir,filename=filename,data_type=data_type)
                    else:
                        continue
'''