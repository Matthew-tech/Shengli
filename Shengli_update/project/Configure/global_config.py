# 41 44 45 50
# 地震体文件的基本属性
region_y_s = 4232695
region_x_s = 640650
region_y_e = 4249270
region_x_e = 681650
cdp_s = 1189
cdp_e = 1852
line_s = 627
line_e = 2267
sampling_points = 1251
delta_x = (region_x_e - region_x_s) / (line_e - line_s)
delta_y = (region_y_e - region_y_s) / (cdp_e - cdp_s)
'''
x: 6*
y: 4*
line:   x
cdp:    y
'''
import platform
import os

OperationSystem = platform.system()
if OperationSystem == 'Windows':
    path_j = '\\'
elif OperationSystem == 'Linux':
    path_j = '/'
SEISMIC_TRACE_SAMPLING = 1251  # 一共有1251个采样点
SEISMIC_TRACE_TIME = 2500  # 一个道数据一共有2500毫秒，每 2ms 进行一次采样

SHUFFLE_ALL_WELLS = True  # 是否将所有的井打乱
SHUFFLE_TEST_SAMPLES = True  # 是否要将所有的测试样本打乱
CHOOSE_BIG_WINDOW_FOR_TEST = True  # 是否选择等分的大时窗进行测试

selected_attr_num = 15  # 选择的地震体的数量  0 < selected_attr_num < 76
trace_range = 1
line_skip = 100


class file_loc_global():  # 地震数据，井数据，油性数据存储位置
    def __init__(self):
        # 预处理过程需要的文件或文件夹
        self.saveFilePath_Base = 'data/full_train_data/'  # 修改
        self.well_data_dir = 'data/1-well'
        self.well_loc_file = os.path.join(self.well_data_dir, 'well_location_new.csv')
        self.seismic_sgy_file_path_base = self.get_seismic_data_filepath()  # 修改
        self.full_train_data = 'data/full_train_data'  # 'data/full_train_data/'修改
        self.well_post_data = os.path.join(self.well_data_dir, 'post_data')
        self.depth_time_rel_dir = os.path.join(self.well_data_dir, 'depth_time_rel')
        self.Target_segment_per_well = os.path.join(self.well_data_dir, 'Target_segment_per_well_zhu.csv')
        self.oil_data_dir = os.path.join(self.well_data_dir, "oil_data")
        self.training_data_dir = 'data/4-training_data'  # 修改
        # 时间表示的顶底文件
        self.bottom_top_file_dir = 'data/2-bottom_top_files/'

        # 全部的有标记的数据
        self.well_reservoir_Info_all = os.path.join(self.well_post_data, 'well_reservoir_rock_oil_merged.csv')
        # 全部的有标记的数据 - 去掉断层
        self.well_reservoir_Info_all_clean = os.path.join(self.well_post_data,
                                                          'well_reservoir_rock_oil_merged_clean.csv')

        # 目标层段生成的储层信息，每个深度对应的标记
        self.well_reservoir_Info = os.path.join(self.well_post_data, 'well_reservoir_Info.csv')
        # 目标层段清除中间的断层
        self.well_reservoir_Info_clean = os.path.join(self.well_post_data, 'well_reservoir_Info_clean.csv')
        self.well_reservoir_rock_oil_merged_target_segment = os.path.join(self.well_post_data,
                                                                          'well_reservoir_rock_oil_merged_target_segment.csv')
        # 新增的label标记文件
        self.supplement_dir = os.path.join(self.well_data_dir, 'litho_res')
        self.high_correlation_saved_file = os.path.join(self.full_train_data, 'high_correlation_attrs.pkl')
        self.correlation_file = os.path.join(self.full_train_data, 'reservoir_seismic_correlation.csv')
        self.infopresent = 'InfoPresent/'

    def get_seismic_data_filepath(self):
        if os.path.exists('/disk3/zk/aboutoil/Shengli/Shengli_update/project/Configure/seismic_data_path.txt'):
            with open('/disk3/zk/aboutoil/Shengli/Shengli_update/project/Configure/seismic_data_path.txt','r') as file:
                return file.readline()
        else:
            print('%s:不存在！'%'Configure/seismic_data_path.txt')
file_loc_gl = file_loc_global()

if __name__ == '__main__':
    print(file_loc_gl.seismic_sgy_file_path_base)