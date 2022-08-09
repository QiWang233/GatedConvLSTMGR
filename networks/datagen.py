# -*- coding: utf-8 -*-
# @Time : 2022/8/9 9:35
# @File : datagen.py
# @Software: PyCharm
# @Author : @white233

import networks.inputs as data
import numpy as np
import threading
from tensorflow.python import keras


# Iteration
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    else:
        indices = None

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# Threading
def threading_data(data=None, fn=None, **kwargs):
    # define function for threading
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    # start multi-threaded reading.
    results = [None] * len(data)  # preallocate result list
    threads = []
    for i in range(len(data)):
        t = threading.Thread(
            name='threading_and_return',
            target=apply_fn,
            args=(results, i, data[i], kwargs)
        )
        t.start()
        threads.append(t)

    # <Milo> wait for all threads to complete
    for t in threads:
        t.join()

    return np.asarray(results)


def DiyGesTrainImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
    X_train, y_train = data.load_diy_video_list(filepath)
    X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
    y_train = np.asarray(y_train, dtype=np.int32)
    while 1:
        for X_indices, y_label_t in minibatches(X_tridx, y_train,
                                                batch_size, shuffle=True):
            # Read data for each batch
            image_path = []
            image_fcnt = []
            image_olen = []
            image_start = []
            is_training = []
            for data_a in range(batch_size):
                X_index_a = X_indices[data_a]
                key_str = '%06d' % X_index_a
                image_path.append(X_train[key_str]['videopath'])
                image_fcnt.append(X_train[key_str]['framecnt'])
                image_olen.append(seq_len)
                image_start.append(1)
                is_training.append(True)  # Training
            image_info = zip(image_path, image_fcnt, image_olen, image_start, is_training)

            X_data_t = threading_data([_ for _ in image_info], data.prepare_diy_depth_data)

            y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
            yield X_data_t, y_hot_label_t


def DiyGesTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
    X_test, y_test = data.load_diy_video_list(filepath)
    X_tridx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    while 1:
        for X_indices, y_label_t in minibatches(X_tridx, y_test,
                                                batch_size, shuffle=True):
            # Read data for each batch
            image_path = []
            image_fcnt = []
            image_olen = []
            image_start = []
            is_training = []
            for data_a in range(batch_size):
                X_index_a = X_indices[data_a]
                key_str = '%06d' % X_index_a
                image_path.append(X_test[key_str]['videopath'])
                image_fcnt.append(X_test[key_str]['framecnt'])
                image_olen.append(seq_len)
                image_start.append(1)
                is_training.append(False)  # Training
            image_info = zip(image_path, image_fcnt, image_olen, image_start, is_training)

            X_data_t = threading_data([_ for _ in image_info], data.prepare_diy_depth_data)

            y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
            yield X_data_t, y_hot_label_t


