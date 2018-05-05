"""
author: Eric
update time:2017.07.26
Description: get well reservoir information from rock files and oil files
"""
import xlrd
import csv
import os
from Configure.global_config import *
files = file_loc_global()
def del_extra_record(record_sorted):
    record_re = []
    for record in record_sorted:
        if record not in record_re:
            record_re.append(record)
    record_no = 1
    for record in record_re:
        record.append(record_no)
        record_no+=1
    return record_re
def convert_to_concise_file(sourceFilePath,DesFilePath,key):
    well_files = os.listdir(sourceFilePath)
    with open(DesFilePath, 'w',newline='') as file:
        csv_writer = csv.writer(file,dialect='excel')
        #writer.writerow(['well_no', 'bottom_depth', 'top_depth', 'is_reservoir', 'reservoir_no'])
        first_line = ['well_no', 'bottom_depth','top_depth','is_reservoir','reservoir_no','差值']
        csv_writer.writerow(first_line)
        Cur_reservoir_type = key_words[key]
        for sourceFileName in well_files:
            sourceFile = os.path.join(sourceFilePath,sourceFileName)
            #print(sourceFile)
            bk = xlrd.open_workbook(sourceFile)
            shxrange = range(bk.nsheets)
            try:
                sh = bk.sheet_by_name("Sheet1")
            except:
                print( "no sheet in %s named Sheet1" % 'Sheet1')
            # 获取行数
            nrows = sh.nrows
            # 获取列数
            ncols = sh.ncols
            # 获取第一行第一列数据
            cell_value = sh.cell_value(1, 1)
            reservoir_no = 0
            all_reservoir_Info = []
            for i in range(1,nrows):
                row_data = sh.row_values(i)
                #print(row_data[Cur_reservoir_type[2]])
                if Cur_reservoir_type[0] in row_data[Cur_reservoir_type[2]].upper() or \
                                Cur_reservoir_type[1] in row_data[Cur_reservoir_type[2]].upper():
                    is_reservoir = 1
                else:
                    is_reservoir = 0
                reservoir_no +=1
                # 分别是  井名称 - 段顶 - 段底 - 是否储层 - 储层号
                Cur_bottom_top_couple = [sourceFileName[:-5],row_data[2], row_data[3],is_reservoir]
                all_reservoir_Info.append(Cur_bottom_top_couple)
            # 按照段顶深度进行排序
            all_reservoir_Info_sorted = sorted(all_reservoir_Info,key=lambda x:x[1])
            # 去除里面的重复记录
            all_reservoir_Info_sorted_del = del_extra_record(all_reservoir_Info_sorted)
            for reservoir_no in range(len(all_reservoir_Info_sorted_del)):
                reservoir_Info = all_reservoir_Info_sorted_del[reservoir_no]
                # 如果不是最后一条记录
                if reservoir_no!= len(all_reservoir_Info_sorted_del)-1:
                    # append 深度差
                    reservoir_Info.append(all_reservoir_Info_sorted_del[reservoir_no+1][1]-reservoir_Info[2])
                else:
                    reservoir_Info.append(0)
                csv_writer.writerow(reservoir_Info)


def convert_to_concise():
    well_data_dir = files.well_data_dir
    rock_data_dir = os.path.join(well_data_dir, 'rock_data')
    oil_data_dir = os.path.join(well_data_dir, 'oil_data')
    global key_words
    #    key_words = {'rock': ['砂', '砾', 9], 'oil': ['油', '水', 6]}   # 错误的标记方法
    key_words = {'rock': ['S', 'L', 8], 'oil': ['油', '水', 6]}
    well_data_processed_dir = os.path.join(well_data_dir, 'post_data')
    if not os.path.exists(well_data_processed_dir):
        os.mkdir(well_data_processed_dir)
    # 生成深度表示的储层文件，1表示是储层，0表示不是储层
    convert_to_concise_file(rock_data_dir,
                            os.path.join(well_data_processed_dir, 'well_reservior_rock_data_sorted_single_attr.csv'),
                            key='rock')
    convert_to_concise_file(oil_data_dir,
                            os.path.join(well_data_processed_dir, 'well_reservior_oil_data_sorted_single_attr.csv'),
                            key='oil')
if __name__ == '__main__':
    convert_to_concise()