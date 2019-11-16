import os
from data_prepare.point_to_label.get_input_data_p2l import get_files_list
from Configure.global_config import file_loc_gl
feature_file_dir = file_loc_gl.full_train_data
attr_count = 0

attr_file_list = get_files_list(feature_file_dir)
for Cur_file in sorted(attr_file_list, key=lambda x: x[0][x[0].index(x[2])+len(x[2])+1:]):
    print(attr_count,os.path.join( Cur_file[0][Cur_file[0].index('3')+2:-4]),'\n')
    attr_count += 1