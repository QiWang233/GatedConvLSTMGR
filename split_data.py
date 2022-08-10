# -*- coding: utf-8 -*-
# @Time : 2022/8/9 10:31
# @File : split_data.py
# @Software: PyCharm
# @Author : @white233
import os
import random

import numpy as np

Windows = 0
Ubuntu = 1
device = Windows

if device == Windows:
    Alldata_path_out_files = "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR"
    Alldata_path_in_files = "dataset_splits\\DiyGD"

    train_list_path = "train_depth_list.txt"
    test_list_path = "test_depth_list.txt"
    valid_list_path = "valid_depth_list.txt"

    data_root = os.path.join(os.getcwd(), "datasets")
else:
    Alldata_path_out_files = "/home/wq"
    Alldata_path_in_files = "GatedConvLSTM/dataset_splits/DiyGD"

    train_list_path = "train_depth_list.txt"
    test_list_path = "test_depth_list.txt"
    valid_list_path = "valid_depth_list.txt"

    data_root = os.path.join(Alldata_path_out_files, 'dataset')

train_percent = 0.6
test_percent = 0.2
valid_percent = 0.2

video_list = os.listdir(data_root)
video_num = len(video_list)
print(video_num * test_percent)


#   write to txt
def Generate_diy_video_list():
    assert os.path.exists(os.path.join(Alldata_path_out_files, Alldata_path_in_files)), "路径错误"
    video_data = {}
    video_label = []
    out_file_root = os.path.join(Alldata_path_out_files, Alldata_path_in_files)

    train_list_file = os.path.join(out_file_root, train_list_path)
    test_list_file = os.path.join(out_file_root, test_list_path)
    valid_list_file = os.path.join(out_file_root, valid_list_path)

    train_file_os = open(train_list_file, 'w')
    test_file_os = open(test_list_file, 'w')
    valid_file_os = open(valid_list_file, 'w')

    test_list = random.sample(video_list, k=int(video_num * test_percent))
    video_list2 = [x for x in video_list if x not in test_list]
    valid_list = random.sample(video_list2, k=int(video_num * valid_percent))
    train_list = [x for x in video_list2 if x not in valid_list]

    for idx, line in enumerate(video_list):
        video_key = '%06d' % idx
        video_path = os.path.join(data_root, str(line))
        video_data[video_key] = {}
        video_data[video_key]['videopath'] = video_path
        video_data[video_key]['framecnt'] = len(os.listdir(os.path.join(video_path, 'leapImg')))
        video_label_single = int(line[1:3])
        video_label.append(video_label_single)

        if line in train_list:
            train_file_os.write(video_data[video_key]['videopath'] + ' '
                                + str(video_data[video_key]['framecnt']) + ' '
                                + str(video_label_single) + '\n')
        elif line in valid_list:
            valid_file_os.write(video_data[video_key]['videopath'] + ' '
                                + str(video_data[video_key]['framecnt']) + ' '
                                + str(video_label_single) + '\n')
        elif line in test_list:
            test_file_os.write(video_data[video_key]['videopath'] + ' '
                               + str(video_data[video_key]['framecnt']) + ' '
                               + str(video_label_single) + '\n')
    train_file_os.close()
    test_file_os.close()
    valid_file_os.close()


Generate_diy_video_list()
