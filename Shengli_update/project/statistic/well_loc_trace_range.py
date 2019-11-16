'''
Date: 2017.08.30
Auther: Eric
Description: 统计每个井对应坐标的目标层段的长度
'''
import matplotlib.pyplot as plt
import csv
import os
from DLmodel.model.Config import files
import pickle

def get_time_range_add_reservoirs():
    time_range_add_reservoirs = {}
    Cur_value = []
    line_last = []
    with open('../'+files.well_reservoir_Info_clean,'r') as file:
        reader = csv.reader(file)
        line_count = 0
        for line in reader:
            if line[0] == 'well_no':
                continue
            well_iter = line[0]
            if line_count == 0:             # 遍历的第一行
                well_Cur = well_iter
                Cur_value.append(float(line[1]))
                line_last = line
            else:
                if well_iter == well_Cur:   # 如果遍历的是同一口井
                    line_last = line
                else:           # 如果遍历到一口新井
                    Cur_value.append(float(line_last[2]))
                    time_range_add_reservoirs.update({well_Cur.upper():Cur_value[1]-Cur_value[0]})
                    line_last = line
                    well_Cur = line[0]
                    Cur_value = [float(line[1])]
            line_count += 1
        # 最后一行
        Cur_value.append(float(line_last[2]))
        time_range_add_reservoirs.update({well_Cur.upper():Cur_value[1]-Cur_value[0]})
    return time_range_add_reservoirs

def main():
    well_name_list = []
    range_list = []
    # 读取每个井对应的时间范围
    with open('../'+files.Target_segment_per_well,'r') as file:
        reader = csv.reader(file)
        well_name_range_dict = {}
        for line in reader:
            if line[0] == 'well_name':
                continue
            well_name_list.append(line[0])
            range_list.append((float(line[2])- float(line[1])))
            well_name_range_dict.update({line[0]:float(line[2])- float(line[1])})

    # 读取每个井被划分成了多少个大时窗
    with open(os.path.join('..',files.train_data_dir,files.attr_file_2),'rb') as file:
        data = pickle.load(file)
        well_name_list_pkl = []
        big_windows_num = []
        wellname_bigw_num_dict = {}
        for well_no in range(len(data)):
            well_name_list_pkl.append(data[int(well_no)][0])
            big_windows_num.append(len(data[well_no][1][0]))
            wellname_bigw_num_dict.update({data[int(well_no)][0].upper():len(data[well_no][1][0])})
    # 找出交集key
    dict_get = get_time_range_add_reservoirs() # 有储层信息的顶底深度
    Intersection_dict = {}
    for key in well_name_range_dict.keys():
        if key in wellname_bigw_num_dict.keys() and key in dict_get.keys():
            Intersection_dict.update({key:[well_name_range_dict.get(key),wellname_bigw_num_dict.get(key),dict_get.get(key)]})
    statistic = sorted(Intersection_dict.values(),key=lambda x:x[0])
    #print(statistic)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(range(len(statistic)),[s[0] for s in statistic])
    ax1.bar(range(len(statistic)),[s[2] for s in statistic],facecolor='palegreen')
    ax1.set_ylabel('time range for each well')
    ax1.set_title('time range and big windows num for each well')
    ax2 = ax1.twinx()
    ax2.plot(range(len(statistic)),[s[1] for s in statistic],'r')
    ax2.set_ylabel('big windows num')
    ax2.set_xlabel('well no.')
    plt.show()
if __name__ == '__main__':
    main()
