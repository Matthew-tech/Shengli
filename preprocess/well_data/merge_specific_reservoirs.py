import csv
import os
import xlrd
from Configure.global_config import *
files = file_loc_global()
def rm_invalid_data(well_reservoir_dict):
    # 去除油性数据中的无效数据，比如CB19中，油性数据的层顶和层底都是0，需要将这部分数据去除
    well_reservoir_dict_re = {}
    for key in well_reservoir_dict.keys():
        if len(well_reservoir_dict.get(key)) == 1:
            if float(well_reservoir_dict.get(key)[0][1])+float(well_reservoir_dict.get(key)[0][2]) == 0:
                print(key)
        else:
            well_reservoir_dict_re.setdefault(key,well_reservoir_dict.get(key))
    return well_reservoir_dict_re
def turn2dict(data,reservoir_type='rock'):
    '''

    :param data: 所有井的储层情况
    :param reservoir_type:
    :return: dict , (key: well_name, value: reservoir_Info)
    '''
    well_reservoir_dict = {}
    if reservoir_type == 'rock':
        # 文件格式： 井名 - x_axis - y_axis - no.
        well_loc_file_reader = csv.reader(open(files.well_loc_file,encoding='utf-8'))
        global well_names_rock
        well_names_rock = []
        for row in well_loc_file_reader:
            well_names_rock.append(row[0])
        # well_names_rock 存放着所有井名
        well_names_rock = well_names_rock[1:]
        # print(len(well_names),well_names)
        for well in well_names_rock:
            well_reservoir_dict.setdefault(well,[reservoir_list for reservoir_list in data if reservoir_list[0] == well])
    elif reservoir_type=='oil':
        global well_names_oil
        well_names_oil = []
        oil_files = os.listdir(files.oil_data_dir)
        for file in oil_files:
            well_names_oil.append(file[:-5])
        for well in well_names_oil:
            well_reservoir_dict.setdefault(well,[reservoir_list for reservoir_list in data if reservoir_list[0] == well])
        well_reservoir_dict_rm = rm_invalid_data(well_reservoir_dict)
        return well_reservoir_dict_rm
    return well_reservoir_dict
def sort_as_depth(data): #data 是 dict，需要把其中的每个元素都进行排序
    all_well_reservoir = []
    for key in sorted(data.keys()):
        #print(well,data.get(well))
        well_reservoirs = sorted(data.get(key),key = lambda reservoir_list:float(reservoir_list[1]))
        all_well_reservoir.extend(well_reservoirs)

    return all_well_reservoir

#首先把连续的储层或者非储层进行合并
def get_data(sourceFile_rock,sourceFile_oil):
    # rock_data
    #csv_reader_rock = csv.reader(open(sourceFile_rock,encoding='utf-8'))
    csv_reader_rock = csv.reader(open(sourceFile_rock,'r'))
    record_num = 0
    Cur_rows_rock = []
    for row_iter in csv_reader_rock:
        Cur_rows_rock.append(row_iter)
    reservoir_rock_dict = turn2dict(Cur_rows_rock[1:],'rock')
    print('rock keys:',len(reservoir_rock_dict.keys()))
    # csv_reader_oil = csv.reader(open(sourceFile_oil,'r',encoding='utf-8'))
    csv_reader_oil = csv.reader(open(sourceFile_oil,'r'))
    Cur_rows_oil = []
    for row_iter in csv_reader_oil:
        Cur_rows_oil.append(row_iter)
    reservoir_oil_dict = turn2dict(Cur_rows_oil[1:],'oil')
    print('oil keys:',len(reservoir_oil_dict.keys()))
    reservoir_merged_dict = merge_rock_oil_data(reservoir_rock_dict,reservoir_oil_dict)
    print('merged keys:',len(reservoir_merged_dict.keys()))
    return sort_as_depth(reservoir_merged_dict) #对储层进行排序
def clean_dict(dict):
    null_value_keys = list(dict.keys())[list(dict.values()).index([])]
    print(null_value_keys)
    for key in null_value_keys:
        print(key)
        dict.pop(key)
    return dict
def merge_rock_oil_data(rock_dict,oil_dict):
    rock_dict_keys = list(rock_dict.keys())
    oil_dict_keys = list(oil_dict.keys())
    for oil_key in oil_dict_keys:
        if oil_key in rock_dict_keys:
    #        print('找到重合well')
            rock_dict_value = rock_dict.get(oil_key)
            oil_dict_value = oil_dict.get(oil_key)
            rock_dict_value.extend(oil_dict_value)
            rock_dict.update({oil_key:rock_dict_value})
    return  rock_dict

# 合并连续储层,
def merge_continuous_reservoir(data):
    '''
    :param data: 岩性和油性储层数据, 可能的情况
                 1) 储层是连续的
                 2) 储层之间有重叠部分
                 3) 储层之间有间隔
    :return:  合并后的储层
    '''
    reservoir_merged = []
    reservoir_to_be_merged = data[0]
    # print('reservoir_to_be_merged:',reservoir_to_be_merged)
    count = 0
    for reservoir in data[1:]:
        # 上下紧邻两层是同一个井，并且都是储层或者非储层
        #print(reservoir)
        # test code
        #print(reservoir[1],len(reservoir[1]),reservoir_to_be_merged[2],len(reservoir_to_be_merged[2]))
        # 如果是同一口井

        if reservoir[0] == reservoir_to_be_merged[0]:
            # 如果是相邻储层，并且标记信息相同
            if reservoir[1] <= reservoir_to_be_merged[2] and int(reservoir[3]) == int(reservoir_to_be_merged[3]):
                reservoir_to_be_merged[2] = reservoir[2]
            else:
                # 如果储层不连续或者标记值不相同
                reservoir_merged.append(reservoir_to_be_merged)
                reservoir_to_be_merged = reservoir
        #剩下的所有情况都表示两个储层不能进行归并
        else:
            reservoir_merged.append(reservoir_to_be_merged)
            reservoir_to_be_merged = reservoir
        count += 1
    return reservoir_merged

def changeReservoir(reservoirs,depth):
    for reservoir in reservoirs:
        # 如果是储层并且其厚度 < depth ，设置为非储层
        if int(reservoir[3]) == 1 and (float(reservoir[2])-float(reservoir[1]))<=depth:
            reservoir[3] = 0
    #print(reservoirs[0])
    return merge_continuous_reservoir(reservoirs)
def save_reservoir(reservoir_merged,DesFile,is_supplement=False,col = 5):
    with open(DesFile,'w',newline='') as out_file:
        csv_writer = csv.writer(out_file,dialect='excel')
        csv_writer.writerow(['well_no','bottom_depth','top_depth','is_reservoir','reservoir_no','差值'])
        Cur_well_reservoir_num = 0
        Cur_well_name = reservoir_merged[0][0]
        for row in reservoir_merged:
            # 删掉reservoir_Info 里面的测井（C 结尾）和斜井（X 结尾）
            if row[0][-1] == 'C' or row[0][-1] == 'X':
                continue
            # 表示是同一个井的储层
            if row[0] == Cur_well_name:
                Cur_well_reservoir_num +=1
                if is_supplement:
                    write_row = row
                else:
                    write_row = []
                    write_row.extend(row[:-2])
                write_row.append(Cur_well_reservoir_num)
                csv_writer.writerow(write_row[:col])
            else: # 遇到了新的井
                Cur_well_reservoir_num = 1
                if is_supplement:
                    write_row = row
                else:
                    write_row = []
                    write_row.extend(row[:-1])
                write_row.append(Cur_well_reservoir_num)
                csv_writer.writerow(write_row[:col])
                Cur_well_name = row[0]
def is_overlap_target_segment(line,well_top_bottom):
    '''
    判断当前储层 line 是否与该井对应的目标层段有重合部分
    :param line:    当前储层
    :param well_top_bottom: 当前储层对应的目的层段
    :return:    是否有重合
    '''

    if float(line[2])>well_top_bottom[0] and float(line[1])<well_top_bottom[0]:
        # 上部有重合
        return 'top_overlap'
    elif float(line[2])<=well_top_bottom[1] and float(line[1])>=well_top_bottom[0]:
        # 包含在目标层段里面
        return 'in'
    elif float(line[2])>well_top_bottom[1] and float(line[1])<well_top_bottom[1]:
        # 下部有重合
        return 'bottom_overlap'
    else:
        return 'no_overlap'

def generate_target_segment_reservoir_file(merged_file):
    # 生成每个井顶底对的dict,(目标层段)
    all_well_ts_file_path = files.Target_segment_per_well
    with open(all_well_ts_file_path,'r',encoding='utf-8') as all_well_ts_file:
        ts_data_reader = csv.reader(all_well_ts_file)
        ts_dict = {} # key: well_name, value: [target_bottom,target_top]
        for line in ts_data_reader:
            if line[0] == 'well_name':
                continue
            # 深度顶和深度底(改变了原始文件的列排布)
            ts_dict.update({line[0].upper():[float(line[1]),float(line[2])]})

    ts_DesFile = merged_file[:-4]+'_target_segment.csv'
    ts_file = open(ts_DesFile,'w',newline='')
    ts_csv_writer = csv.writer(ts_file,dialect='excel')
    ts_csv_writer.writerow(['well_no','bottom_depth','top_depth','is_reservoir','reservoir_no'])
    # DesFile 是全部层段信息，遍历其中每一行并判断是否在目标层段内
    merged_file_ori = open(merged_file,'r')
    reader  = csv.reader(merged_file_ori)
    no_ts_well = []         # 保存不在目标层段内的井名称
    exist_ts_well = []      # 保存与目标层段有重合的井的名称
    no_ts_Info = []         # 保存在各个井目标层段文件中不存在的井名称
    line_exe = 0    # 已处理的行数
    Cur_well_reservoir_num = 0 #记录当前井的储层数量
    for line in reader:
        if line[0] == 'well_no':
            continue
        else:
            if line_exe == 0:
                Cur_well_name = line[0].upper()

            iter_well_name = line[0]
            line_exe += 1
            # 判断当前储层是否跟目的层段有重合
            # 返回值有四种，分别是：
            # 'top_overlap':    上部有重合
            # 'in':             嵌入到目标层段
            # 'bottom_overlap': 下部有重合
            # 'no_overlap':     没有重合部分
            ts_range = ts_dict.get(iter_well_name.upper())
            if ts_range is not None:
                overlap_type = is_overlap_target_segment(line,ts_range)
            else:
                # 没有该井的目标层段记录
                no_ts_Info.append(iter_well_name)
                continue
            if overlap_type!='no_overlap':
                # 表示有重叠部分
                exist_ts_well.append(iter_well_name)
                # 根据返回值定义新的重叠层段
                if overlap_type == 'top_overlap':
                    overlap_segment = [line[0],ts_dict.get(iter_well_name)[0],float(line[2]),line[3]]
                elif overlap_type == 'in':
                    overlap_segment = line[:-1]
                else:   # 表示下部有重合
                    overlap_segment = [line[0],float(line[1]),ts_dict.get(iter_well_name)[1],line[3]]
                # 表示是同一个井
                if iter_well_name.upper() == Cur_well_name.upper():
                    Cur_well_reservoir_num += 1
                    write_row = []
                    write_row.extend(overlap_segment)    #去掉最后一个储层序号
                    write_row.append(Cur_well_reservoir_num)    #补上一个新的储层序号
                    ts_csv_writer.writerow(write_row)
                # 如果是另外一口井
                else:
                    Cur_well_reservoir_num = 1
                    write_row = []
                    write_row.extend(overlap_segment)
                    write_row.append(Cur_well_reservoir_num)
                    ts_csv_writer.writerow(write_row)
                    Cur_well_name = line[0].upper()
    # exist_ts_well 记录与目标层段有重合的井
    exist_ts_well = set(exist_ts_well)
    for well_name in ts_dict.keys():
        if well_name not in exist_ts_well and well_name.upper() not in exist_ts_well and well_name.lower() not in exist_ts_well:
            # 记录与目标层段没有重合的井
            no_ts_well.append(well_name)
    print('no:',len(no_ts_well),no_ts_well,)
    print('exist:',len(exist_ts_well),exist_ts_well,)
    print('no ts Info:',set(no_ts_Info))
    ts_file.close()
    merged_file_ori.close()
def generate_reservoir_Info(DesFile):
    sourceFileBase = file_loc_gl.well_post_data
    files = ['well_reservior_rock_data_sorted_single_attr.csv','well_reservior_oil_data_sorted_single_attr.csv']
    # all_data 是所有的岩性储层和油性储层信息
    all_data = get_data(os.path.join(sourceFileBase,files[0]), os.path.join(sourceFileBase,files[1]))
    well_name_list = [r[0] for r in all_data]
    reservoir_merged = merge_continuous_reservoir(all_data)
    save_reservoir(reservoir_merged, DesFile)


def merge_reservoirs():
    sourceFileBase = files.well_post_data
    # 合并岩石属性和油气属性，并把相邻的同层段进行合并
    DesFile = os.path.join(sourceFileBase, 'well_reservoir_rock_oil_merged.csv')
    if not os.path.exists(DesFile):
        generate_reservoir_Info(DesFile)

    merged_file = DesFile
    generate_target_segment_reservoir_file(merged_file)


if __name__ == '__main__':
    merge_reservoirs()