# -*- coding: utf-8 -*-
# @Time : 2022/8/2 10:43
# @File : inputs.py
# @Software: PyCharm
# @Author : @white233

import os
import numpy as np
import math
import random
from scipy.misc import imread, imresize


def load_diy_video_list(path):
    assert os.path.exists(path)
    f = open(path, 'r')
    f_lines = f.readlines()
    f.close()
    video_data = {}
    video_label = []
    for idx, line in enumerate(f_lines):
        video_key = '%06d' % idx
        video_data[video_key] = {}
        videopath = line.split(' ')[0]  # path
        framecnt = int(line.split(' ')[1])  # 32
        videolabel = int(line.split(' ')[2])  # class
        video_data[video_key]['videopath'] = videopath
        video_data[video_key]['framecnt'] = framecnt
        video_label.append(videolabel)
    return video_data, video_label


def prepare_diy_depth_data(image_info):
    video_path = image_info[0]
    video_frame_cnt = image_info[1]
    output_frame_cnt = image_info[2]
    start_frame_idx = image_info[3]
    is_training = image_info[4]
    assert os.path.exists(video_path)
    rand_frames = np.zeros(output_frame_cnt)
    div = float(video_frame_cnt) / float(output_frame_cnt)
    scale = math.floor(div)
    if is_training:
        if scale == 0:
            rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
            rand_frames[video_frame_cnt::] = video_frame_cnt - 1
        elif scale == 1:
            rand_frames[::] = div * np.arange(0, output_frame_cnt)
        else:
            rand_frames[::] = div * np.arange(0, output_frame_cnt) + \
                              float(scale) / 2 * (np.random.random(size=output_frame_cnt) - 0.5)
    else:
        if scale == 0:
            rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
            rand_frames[video_frame_cnt::] = video_frame_cnt - 1
        else:
            rand_frames[::] = div * np.arange(0, output_frame_cnt)
    rand_frames[0] = max(rand_frames[0], 0)
    rand_frames[output_frame_cnt - 1] = min(rand_frames[output_frame_cnt - 1], video_frame_cnt - 1)
    rand_frames = np.floor(rand_frames)

    average_values = [127]
    processed_images = np.empty((output_frame_cnt, 112, 112, 1), dtype=np.float32)
    crop_random = random.random()
    for idx in range(0, output_frame_cnt):
        # E:\pycharm\PycharmProjects\GatedConvLSTMGR\datasets\C00S0001A20\leapImg\000.png
        image_file = '%s\\leapImg\\%03d.png' % (video_path, rand_frames[idx])
        assert os.path.exists(image_file)
        image = imread(image_file)
        image_h, image_w = np.shape(image)
        square_sz = min(image_h, image_w)
        if is_training:
            crop_h = int((image_h - square_sz) * crop_random)
            crop_w = int((image_w - square_sz) * crop_random)
        else:
            crop_h = int((image_h - square_sz) / 2)
            crop_w = int((image_w - square_sz) / 2)
        image_crop = image[crop_h:crop_h + square_sz, crop_w:crop_w + square_sz]
        processed_images[idx] = np.reshape((imresize(image_crop, (112, 112)) - average_values), (112, 112, 1))
    return processed_images

