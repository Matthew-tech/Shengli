import csv
from data_prepare.point_to_label.data_util import Build_Interpolation_function
import os
from preprocess.well_data.merge_specific_reservoirs import merge_continuous_reservoir,save_reservoir
from decimal import Decimal
from Configure.global_config import *
files = file_loc_global()
def Turn_to_depth(well_dt_path,well_name, data):
    '''
    :param data: 时间表示的储层信息
    :return:     通过时深转换改成深度表示的储层信息
    '''
    depth_list, time_list, f= Build_Interpolation_function(os.path.join(well_dt_path,well_name+'_ck.txt'),is_reverse=True)
    if depth_list == -1:    # 时深关系文件不存在
        depth_list, time_list, f = Build_Interpolation_function(os.path.join(well_dt_path, well_name.upper() + '_ck.txt'),
                                                                is_reverse=True)
        if depth_list == -1:
            return -1
    if depth_list!=-1:
        data_depth = [[iter[0].upper(),float(Decimal(f(iter[1]).tolist()).quantize(Decimal('0.000'))),
                       float(Decimal(f(iter[2]).tolist()).quantize(Decimal('0.000'))),iter[3]] for iter in data]
        return data_depth

def add_supplement_labels():
    depth_time_rel_dir = files.depth_time_rel_dir
    supplement_dir = files.supplement_dir
    # 读入所有的新增label，并将其时间表示转化为深度表示
    supplement_reservoir_Info = []
    for file in sorted(os.listdir(supplement_dir)):
        print(os.path.join(supplement_dir,file))
        with open(os.path.join(supplement_dir,file),'r') as each_file:
            lines = csv.reader(each_file)      # time - is_reservoir
            Cur_well_Info = []
            for line in lines:
                if line[0] == 'time':
                    continue
                Cur_Info = [file[:-7],float(line[0]),float(line[0])+2,int(line[1])]
                Cur_well_Info.append(Cur_Info)
            # 合并连续的储层或者非储层
            Cur_well_Info_fragment = merge_continuous_reservoir(Cur_well_Info)
            Cur_well_Info_depth = Turn_to_depth(depth_time_rel_dir,file[:-7],Cur_well_Info_fragment)

            if Cur_well_Info_depth == -1:   # 时深关系文件不存在
                print('时深关系文件不存在')
                continue
        supplement_reservoir_Info.extend(Cur_well_Info_depth)   # 保存了所有的新增label

    # well_name - bottom_depth - top_depth - is_reservoir
    merged_file = files.well_reservoir_rock_oil_merged_target_segment
    with open(merged_file,'r') as m_file:
        m_reader = csv.reader(m_file)       # merged_file
        reservoir_Info = []
        for line in m_reader:
            reservoir_Info.append(line)
        # 合并新增的label标记到旧的label中
        reservoir_Info.extend(supplement_reservoir_Info)
        reservoir_Info.sort(key=lambda x:(x[0],x[1]))

    # 将合并后的label保存到下来
    saved_file = files.well_reservoir_Info
    save_reservoir(reservoir_Info[:-1],saved_file,is_supplement=True,col = 5)   # 只保留前5列
def clean_reservoir_Info(sourceFile = files.well_reservoir_Info, desFile = files.well_reservoir_Info_clean):
    well_reservoir_Info_clean = []      # 保存去掉不连续数据的储层
    line_last = []      # 表示记录的上一行
    with open(sourceFile,'r') as file:
        reader = csv.reader(file)
        line_count = 0
        for line in reader:
            if line[0] == 'well_no':
                well_reservoir_Info_clean.append(line)
                continue
            well_iter = line[0]
            if line_count == 0:             # 遍历的第一行
                well_Cur = well_iter
                well_reservoir_Info_clean.append(line)
                line_last = line
            else:
                if well_iter == well_Cur:   # 如果遍历的是同一口井
                    line[1] = line_last[2]
                    well_reservoir_Info_clean.append(line)

                    line_last = line
                else:           # 如果遍历到一口新井
                    well_reservoir_Info_clean.append(line)
                    line_last = line
                    well_Cur = line[0]
            line_count += 1
    # 生成一个新的文件
    with open(desFile,'w',newline='') as file:
        writer = csv.writer(file,dialect='excel')
        for line in well_reservoir_Info_clean:
            writer.writerow(line)

if __name__ == '__main__':
    add_supplement_labels()
    clean_reservoir_Info()