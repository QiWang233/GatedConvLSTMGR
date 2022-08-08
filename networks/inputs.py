# -*- coding: utf-8 -*-
# @Time : 2022/8/2 10:43
# @File : inputs.py
# @Software: PyCharm
# @Author : @white233

import os


def load_iso_video_list(path):
    assert os.path.exists(path)
    f = open(path, 'r')
    f_lines = f.readlines()
    f.close()
    video_data = {}
    video_label = []
    for idx, line in enumerate(f_lines):
        video_key = '%06d' % idx
        video_data[video_key] = {}
        videopath = line.split(' ')[0]
        framecnt = int(line.split(' ')[1])
        videolabel = int(line.split(' ')[2])
        video_data[video_key]['videopath'] = videopath
        video_data[video_key]['framecnt'] = framecnt
        video_label.append(videolabel)
    return video_data, video_label
