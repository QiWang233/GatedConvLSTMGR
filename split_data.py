# -*- coding: utf-8 -*-
# @Time : 2022/8/9 10:31
# @File : split_data.py
# @Software: PyCharm
# @Author : @white233
import os
import random
from enum import Enum
import argparse
import time
import numpy as np

Windows = 0
Ubuntu = 1
device = Windows

train_list_path = "train_depth_list.txt"
test_list_path = "test_depth_list.txt"
valid_list_path = "valid_depth_list.txt"

if device == Windows:
    Alldata_path_out_files = "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR"
    Alldata_path_in_files = "dataset_splits\\DiyGD"

    data_root = os.path.join(os.getcwd(), "datasets")
else:
    Alldata_path_out_files = "/home/wq"
    Alldata_path_in_files = "GatedConvLSTM/dataset_splits/DiyGD"

    data_root = os.path.join(Alldata_path_out_files, 'dataset')

train_percent = 0.6
test_percent = 0.2
valid_percent = 0.2

video_list = os.listdir(data_root)
video_num = len(video_list)
print(video_num * test_percent)

split_way = {"head": 10,
             "tail": 20,
             "head_tail": 30,
             "whole_body": 40,
             "slow_gs": 5,
             }


def Split_list_with_diff_way(txt_list_path, way, out_list_path):
    """
    Split data in different way by C__S__Axx
    """
    assert os.path.exists(txt_list_path), "txt 路径不存在！！"

    txt_file_io = open(txt_list_path, 'r')
    out_list_path_io = open(out_list_path, 'w')

    out_list_path_io.seek(0)  # 定位
    out_list_path_io.truncate()  # 清空文件
    f_lines = txt_file_io.readlines()
    txt_file_io.close()

    for line in f_lines:
        if int(line.split(' ')[0].split('/')[4].split('A')[1]) == way:
            out_list_path_io.write(line)
        elif int(line.split(' ')[0].split('/')[4].split('A')[1][0]) == way:
            out_list_path_io.write(line)

    out_list_path_io.close()


def Split_list_into_rvt(txt_list_path, out_list_path, way, if_clear=True):
    """
    split special txt data into train/valid/test
    """
    assert os.path.exists(txt_list_path), "txt 路径不存在！！"

    txt_list_io = open(txt_list_path, 'r')
    f_lines = txt_list_io.readlines()
    line_num = len(f_lines)
    txt_list_io.close()

    train_list_path_here = "%s_train_depth_list.txt" % way
    test_list_path_here = "%s_test_depth_list.txt" % way
    valid_list_path_here = "%s_valid_depth_list.txt" % way

    train_list_file = os.path.join(out_list_path, train_list_path_here)
    test_list_file = os.path.join(out_list_path, test_list_path_here)
    valid_list_file = os.path.join(out_list_path, valid_list_path_here)
    train_file_io = open(train_list_file, 'w')
    test_file_io = open(test_list_file, 'w')
    valid_file_io = open(valid_list_file, 'w')

    if if_clear:
        train_file_io.seek(0)  # 定位
        train_file_io.truncate()  # 清空文件
        test_file_io.seek(0)  # 定位
        test_file_io.truncate()  # 清空文件
        valid_file_io.seek(0)  # 定位
        valid_file_io.truncate()  # 清空文件

    test_list = random.sample(f_lines, k=int(line_num * test_percent))
    f_lines2 = [x for x in f_lines if x not in test_list]
    valid_list = random.sample(f_lines2, k=int(line_num * valid_percent))
    train_list = [x for x in f_lines2 if x not in valid_list]

    for line in f_lines:
        if line in train_list:
            train_file_io.write(line)
        elif line in valid_list:
            valid_file_io.write(line)
        elif line in test_list:
            test_file_io.write(line)

    train_file_io.close()
    test_file_io.close()
    valid_file_io.close()
    print("separate %s into rvt" % txt_list_path)


def Generate_diy_video_list(out_path, video_ls):
    """
    connect the list and list in the txt
    """
    video_data = {}
    video_label = []
    out_file = open(out_path, 'w')
    for idx, line in enumerate(video_ls):
        video_key = '%06d' % idx
        video_path = os.path.join(data_root, str(line))
        video_data[video_key] = {}
        video_data[video_key]['videopath'] = video_path
        video_data[video_key]['framecnt'] = len(os.listdir(os.path.join(video_path, 'leapImg')))
        video_label_single = int(line[1:3])
        video_label.append(video_label_single)
        out_file.write(video_data[video_key]['videopath'] + ' '
                       + str(video_data[video_key]['framecnt']) + ' '
                       + str(video_label_single) + '\n')
    out_file.close()


def All_diy_video_list_split():
    """
    read && make video list into train/valid/test.txt three part together
    """
    out_file_root = os.path.join(Alldata_path_out_files, Alldata_path_in_files)
    assert os.path.exists(out_file_root), "路径错误"
    video_data = {}
    video_label = []

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


def Mix_list_into_rvt(txt_list_path_root, proportion, out_txt_name):
    """
    proportion : 8_1_1  or 3_5_1_1 ....
    """
    txt_file_list = [way for way in str(proportion).split(',')[0].split('_')]
    proportion_list = [way for way in str(proportion).split(',')[1].split('_')]
    print(txt_file_list)
    # print(proportion_list)

    proportion_list = [float(proportion) for proportion in proportion_list]
    proportion_sum = sum(proportion_list)
    proportion_list = [float(proportion / proportion_sum) for proportion in proportion_list]
    print(proportion_list)

    out_io = open(os.path.join(txt_list_path_root, out_txt_name), 'w')
    out_io.seek(0)  # 定位
    out_io.truncate()  # 清空文件

    origin_file_io = {}
    f_origin_file_io = {}
    all_prefix = {}
    for idx, way in enumerate(txt_file_list):
        origin_file_io[idx] = open(os.path.join(txt_list_path_root, '%s_origin_depth_list.txt' % way), 'r')
        f_origin_file_io[idx] = origin_file_io[idx].readlines()
        all_prefix[idx] = []
        origin_file_io[idx].close()
    num_for_all = len(f_origin_file_io[0])
    print('10_origin_depth_list.txt_num:%d ' % num_for_all)
    # print(all_prefix)
    total_prefix = []
    recovery_list = {}
    for i in range(0, num_for_all):
        total_prefix.append(f_origin_file_io[0][i].split(' ')[0].split('/')[4].split('A')[0])
    print(len(total_prefix))

    for t in range(0, len(txt_file_list)):
        if t != 0:
            for x in all_prefix[t - 1]:
                if x in total_prefix:
                    total_prefix.remove(x)
        print('%d:%d' % (t, len(total_prefix)))

        if t == len(txt_file_list) - 1:
            all_prefix[t] = total_prefix
        else:
            all_prefix[t] = random.sample(total_prefix, k=int(num_for_all * proportion_list[t]))
            print(num_for_all * proportion_list[t], int(num_for_all * proportion_list[t]), len(all_prefix[t]))

        recovery_list[t] = [s + 'A%s' % txt_file_list[t] for s in all_prefix[t]]

        for single in recovery_list[t]:
            out_io.write('/home/wq/dataset' + '/' + single + ' 32 ' + single[2:3] + '\n')
    out_io.close()

    if device == Windows:
        copy_txt(os.path.join(Alldata_path_out_files, 'dataset_splits\\DiyGD\\5_depth_list.txt'),
                 'E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\%s_mix_depth_list.txt' %
                 (str(proportion).split(',')[0] + '_' + str(proportion).split(',')[1])
                 )
    else:
        copy_txt(os.path.join(os.path.join(Alldata_path_out_files, Alldata_path_in_files), '5_depth_list.txt'),
                 os.path.join(os.path.join(Alldata_path_out_files, Alldata_path_in_files), '%s_mix_depth_list.txt') %
                 (str(proportion).split(',')[0] + '_' + str(proportion).split(',')[1])
                 )

    Split_list_into_rvt(os.path.join(txt_list_path_root, out_txt_name), txt_list_path_root,
                        str(proportion).split(',')[0] + '_' + str(proportion).split(',')[1])


def copy_txt(a, b):
    text = open(a, 'r')
    txt = open(b, 'a')
    text.seek(0)
    f = text.readlines()
    for line in f:
        txt.write(line)
    txt.close()
    text.close()


if __name__ == '__main__':
    # WAY = split_way["slow_gs"]

    parser = argparse.ArgumentParser(description='split data way')
    parser.add_argument('--wp', type=str, help='such as \'10_20_30,3_2_5\'')
    args = parser.parse_args()

    if device == Windows:
        Mix_list_into_rvt(os.path.join(Alldata_path_out_files, Alldata_path_in_files), args.wp,
                          '%s_mix_depth_list.txt' % (str(args.wp).split(',')[0] + '_' + str(args.wp).split(',')[1]))
    else:
        Mix_list_into_rvt(os.path.join(Alldata_path_out_files, Alldata_path_in_files), args.wp,
                          '%s_mix_depth_list.txt' % (str(args.wp).split(',')[0] + '_' + str(args.wp).split(',')[1]))

    # with open('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\5_depth_list.txt', 'r') as text:
    #     with open('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\mix_depth_list.txt', 'a') as txt:
    #         txt.writelines(text.readlines())

    # for i in [value for value in split_way.values()]:
    #     Split_list_with_diff_way(
    #         "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\all_depth_list.txt",
    #         i, "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD"
    #            "\\%d_origin_depth_list.txt" % i)

    # for i in [value for value in split_way.values()]: Split_list_into_rvt(
    # "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\%s_depth_list.txt" % i,
    # "E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD", i)

    # for i in [value for value in split_way.values()]: with open(
    # 'E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\5_depth_list.txt', 'r') as text: with
    # open('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\%d_depth_list.txt' % i,
    # 'a') as txt: txt.writelines(text.readlines())

    # with open('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\5_depth_list.txt',
    # 'r') as text: with open('E:\\pycharm\\PycharmProjects\\GatedConvLSTMGR\\dataset_splits\\DiyGD\\10_depth_list
    # .txt', 'a') as txt: txt.writelines(text.readlines())
x= np.zeros()