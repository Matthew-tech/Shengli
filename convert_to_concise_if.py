# -*- coding: utf-8 -*-
'''
井数据处理
1.输入：data_dir:岩石数据位置， key：'rock'
2.输入：data_dir:油性数据位置， key：'oil'
'''
import os
#print(os.getcwd())
import sys
#print(sys.path)
from preprocess.well_data.convert_reservoir_to_concise import *


def convert_to_concise_file(sourceFilePath,DesFilePath,key):
    well_files = os.listdir(sourceFilePath)
    with open(DesFilePath, 'w',newline='') as file:
        csv_writer = csv.writer(file,dialect='excel')
        #writer.writerow(['well_no', 'bottom_depth', 'top_depth', 'is_reservoir', 'reservoir_no'])
        first_line = ['well_no', 'bottom_depth','top_depth','is_reservoir','reservoir_no','差值']
        csv_writer.writerow(first_line)
        Cur_reservoir_type = key_words[key]
        for sourceFileName in well_files:
            #sourceFileName = "CB111.xlsx"
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
            # print("nrows %d, ncols %d" % (nrows, ncols))
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
                #print(Cur_bottom_top_couple)
                #results+=sourceFileName[:-5]+','+str(Cur_bottom_top_couple[1:])[1:-1]+'\n'
                #writer.writerow(Cur_bottom_top_couple)


def convert_to_concise_if(data_dir, key):
    well_data_dir = files.well_data_dir
    # rock_data_dir = os.path.join(well_data_dir, 'rock_data')
    # oil_data_dir = os.path.join(well_data_dir, 'oil_data')
    global key_words
    #    key_words = {'rock': ['砂', '砾', 9], 'oil': ['油', '水', 6]}   # 错误的标记方法
    key_words = {'rock': ['S', 'L', 8], 'oil': ['油', '水', 6]}
    well_data_processed_dir = os.path.join(well_data_dir, 'post_data')
    if not os.path.exists(well_data_processed_dir):
        os.mkdir(well_data_processed_dir)
    # 生成深度表示的储层文件，1表示是储层，0表示不是储层
    convert_to_concise_file(data_dir,
                            os.path.join(well_data_processed_dir, 'well_reservior_' + key + '_data_sorted_single_attr.csv'),
                            key=key)


if __name__ == "__main__":
    convert_to_concise_if(sys.argv[1], sys.argv[2])  # 选择岩性数据／油性数据
    #convert_to_concise_if('/home/eric/Desktop/Shengli_update_clean/project/data/1-well/oil_data', 'oil')
