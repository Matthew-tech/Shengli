'''
Description:    统计所有位置的顶底时间差，并把所有坐标对应的顶底时间存入pkl中的dict文件，便于之后的读取
Auther:         Eric
Date:           2017.08.25
'''
import os
import pickle
import seaborn as sns
from Configure.global_config import file_loc_gl as files
import matplotlib.pyplot as plt

b_t_dict_file = os.path.join(files.bottom_top_file_dir,'b_t_dict.pkl')
def save_as_dict():
    if os.path.exists(b_t_dict_file):
        print('b_t_dict.pkl文件已存在！！！')
        return
    b_t_dict = {}
    for filename in os.listdir(files.bottom_top_file_dir):
        if '.pkl' in filename:
            continue
        print(filename)
        with open(os.path.join(files.bottom_top_file_dir,filename),'r') as file:
            file.readline()
            file.readline()
            line = file.readline()
            while line:
                line = line.split(' ')
                line = [ele for ele in line if ele !='']
                if 'EOD' in line:
                    break
                key = str(int(float(line[4])))+'.'+str(int(float(line[3])))
                value = float(line[2])
                if key in b_t_dict.keys():
                    Cur_value = b_t_dict.get(key)
                    Cur_value.append(value)
                    Cur_value = sorted(Cur_value)
                    b_t_dict.update({ key: Cur_value})
                else:
                    b_t_dict.update({key:[value]})
                line = file.readline()
    # 将生成的b_t_list 保存到本地
    with open(os.path.join(files.bottom_top_file_dir,'b_t_dict.pkl'),'wb') as file:
        pickle.dump(b_t_dict, file, -1)

def read_b_t_dict():
    if os.path.exists(b_t_dict_file):
        with open(b_t_dict_file,'rb') as file:
            b_t_dict = pickle.load(file)
        return b_t_dict
    else:
        print('b_t_dict.pkl文件不存在！！！')
def main():
    save_as_dict()
    b_t_dict = read_b_t_dict()
    time_range = []
    print('正在进行排序...')
    dict_sorted =  sorted(b_t_dict.items(),key=lambda x:float(x[0]))
    time_range = [ele[1] for ele in dict_sorted]
    '''
    print('正在画柱状统计图...(%g)'%len(time_range))
    for time_range_no in range(len(time_range)):
        plt.plot([time_range_no,time_range_no],time_range[time_range_no],'b')
        if time_range_no%2000 == 0:
            print('%g / %g'%(time_range_no,len(time_range)))
    plt.show()
    '''
    print(time_range[0])
    print('正在统计...')
    statistic_dict = {'a':0,'b':0,'c':0,'d':0}         # 'a': time_range 0~100    'b': time_range 100~200 'c': 200~300, 'd': 300~400  'e':>400
#    plt.hist([t_r[0] for t_r in time_range],bins=20,color='steelblue',normed=True)
    plt.hist([t_r[1]-t_r[0] for t_r in time_range],bins=100,color='steelblue',normed=True)
    plt.title('Distribution of Target Segment Time Range')
    plt.xlabel('Target Segment Time Range')
    plt.ylabel('Partition')
    plt.show('Proportion')
#    sns.distplot([t_r[0] for t_r in time_range],rug=True)
if __name__ == '__main__':
    main()
