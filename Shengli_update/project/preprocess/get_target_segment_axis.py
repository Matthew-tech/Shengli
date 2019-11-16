import csv
import os
def main():
    sourceFile = '..\\data\\1-well\\CDD_T0_top_ffs_grid-n_22mar_171854.p701'
    desFile_1 = '..\\data\\ts_top_1.csv'
    desFile_2 = '..\\data\\ts_top_2.csv'
    if os.path.exists(desFile_1):
        print(desFile_1+'文件已存在...')
        exit()
    with open(sourceFile,'r') as file:
        writer_file_1 = open(desFile_1,'w',newline='')
        writer_1 = csv.writer(writer_file_1,dialect='excel')
        writer_1.writerow(['x_axis','y_axis','top','trace_no','line_no'])
        writer_file_2 = open(desFile_2, 'w', newline='')
        writer_2 = csv.writer(writer_file_2, dialect='excel')
        writer_2.writerow(['x_axis', 'y_axis', 'top', 'trace_no', 'line_no'])
        file.readline()
        file.readline()
        line = file.readline()
        line_num = 0
        while line:
            line_num += 1
            print(line_num)
            line = line.split(' ')
            line = [e for e in line if e!='']
            line[4] = line[4][:-1]
            line_list = [float(e) for e in map(float,line)]
            if line_list[4]<1435:
                writer_1.writerow(line_list)
            else:
                writer_2.writerow(line_list)
            line = file.readline()
        writer_file_1.close()
        writer_file_2.close()
if __name__ == '__main__':
    main()
    