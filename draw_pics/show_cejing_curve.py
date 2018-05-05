import os
import matplotlib.pyplot as plt
import sys
def show_cejing_curve(well_name,depth_list,cejing_curve_list=None,curve_name = 'us/ft'):
    if isinstance(cejing_curve_list,list):
        curve_name_list = ['gAPI','mV_BC','us/ft']
        for curve_no in range(len(cejing_curve_list)):
            plt.subplot(3,1,curve_no+1)
            plt.plot(depth_list,cejing_curve_list[curve_no])
            plt.ylabel(curve_name_list[curve_no])
            if curve_no == 0:
                plt.title(well_name)
            if curve_no == len(cejing_curve_list)-1:
                plt.xlabel('Depth')
        plt.show()
    else:
        plt.plot(depth_list, cejing_curve_list)
        plt.title(well_name)
        plt.xlabel('Depth')
        plt.ylabel(curve_name)
        plt.show()
def main():
    cejing_dir = 'H:\\pg\\2-胜利油田\\数据\\logdata\\sp_jz\\'
    for well_name in os.listdir(cejing_dir):
        well_file = os.path.join(cejing_dir,well_name)
        with open(well_file,'r',encoding='utf-8') as file:
            depth_list = []
            Natural_gamma_curve_list = []   # 自然伽马曲线
            Natural_potential_curve_list = [] # 自然电位曲线
            st_difference_curve_list = []   # 声波时差曲线
            cejing_curve = []
            Cur_line = file.readline()
            Cur_line = file.readline()
            while Cur_line:
                Cur_line = Cur_line.split(' ')
                Cur_line = [Cur_line[i] for i in range(len(Cur_line)) if Cur_line[i]!='' and Cur_line[i]!='\n']
                Cur_line = list(map(float,Cur_line))
                Cur_line = [Cur_line[i] if Cur_line[i]!=-99999 else 0 for i in range(len(Cur_line))]
                depth_list.append(Cur_line[0])
                Natural_gamma_curve_list.append(Cur_line[1])
                Natural_potential_curve_list.append(Cur_line[2])
                st_difference_curve_list.append(Cur_line[3])
                Cur_line = file.readline()
            cejing_curve.append(Natural_gamma_curve_list)
            cejing_curve.append(Natural_potential_curve_list)
            cejing_curve.append(st_difference_curve_list)
        show_cejing_curve(well_name,depth_list,cejing_curve)

        # show_cejing_curve(well_name,depth_list,Natural_gamma_curve_list,'gAPI')
        # show_cejing_curve(well_name,depth_list, Natural_potential_curve_list,'mV_BC')
        # show_cejing_curve(well_name,depth_list,st_difference_curve_list, 'SDC')


if __name__ == '__main__':
    main()