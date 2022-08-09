# -*- coding: utf-8 -*-
# @Time : 2022/8/9 10:31
# @File : split_data.py
# @Software: PyCharm
# @Author : @white233
import os

Windows = 0
Ubuntu = 1

device = Windows

if device == Windows:
    Alldata_path_out_files = "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR"
    Alldata_path_in_files = "dataset_splits/DiyGD/train_depth_list.txt"
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "datasets")
else:
    Alldata_path_out_files = "\\home\\wq"
    Alldata_path_in_files = "dataset_splits/DiyGD/train_depth_list.txt"
    data_root = os.path.join(Alldata_path_out_files, 'dataset')


video_list = os.listdir(data_root)
video_num = len(video_list)


#   write to txt
def Generate_diy_video_list():
    video_data = {}
    video_label = []
    out_file = open(os.path.join(Alldata_path_out_files, Alldata_path_in_files), 'w')
    for idx, line in enumerate(video_list):
        video_key = '%06d' % idx
        video_path = os.path.join(data_root, str(line))
        video_data[video_key] = {}
        video_data[video_key]['videopath'] = video_path
        video_data[video_key]['framecnt'] = len(os.listdir(os.path.join(video_path, 'leapImg')))
        video_label_single = int(line[1:3])
        video_label.append(video_label_single)
        out_file.write(video_data[video_key]['videopath']+' '
                       + str(video_data[video_key]['framecnt'])+' '
                       + str(video_label_single)+'\n')
    out_file.close()






