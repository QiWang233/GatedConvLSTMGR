# -*- coding: utf-8 -*-
# @Time : 2022/8/2 11:14
# @File : res3d_clstm_mobilenet.py
# @Software: PyCharm
# @Author : @white233


import io
import sys

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers, regularizers
l2 = regularizers.l2


def res3d(inputs, weight_decay):
    # Res3D Block 1
    conv3d_1 = layers.Conv3D(64, (3, 7, 7), strides=(1, 2, 2), padding='same',
                             dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_decay), use_bias=False,
                             name='Conv3D_1')(inputs)
    conv3d_1 = layers.BatchNormalization(name='BatchNorm_1_0')(conv3d_1)

    conv3d_1 = layers.Activation('relu', name='ReLU_1')(conv3d_1)

    # Res3D Block 2
    conv3d_2a_1 = layers.Conv3D(64, (1, 1, 1), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_2a_1')(conv3d_1)
    conv3d_2a_1 = layers.BatchNormalization(name='BatchNorm_2a_1')(conv3d_2a_1)
    conv3d_2a_a = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_2a_a')(conv3d_1)
    conv3d_2a_a = layers.BatchNormalization(name='BatchNorm_2a_a')(conv3d_2a_a)

    conv3d_2a_a = layers.Activation('relu', name='ReLU_2a_a')(conv3d_2a_a)

    conv3d_2a_b = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_2a_b')(conv3d_2a_a)
    conv3d_2a_b = layers.BatchNormalization(name='BatchNorm_2a_b')(conv3d_2a_b)

    conv3d_2a = layers.Add(name='Add_2a')([conv3d_2a_1, conv3d_2a_b])
    conv3d_2a = layers.Activation('relu', name='ReLU_2a')(conv3d_2a)

    conv3d_2b_a = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_2b_a')(conv3d_2a)
    conv3d_2b_a = layers.BatchNormalization(name='BatchNorm_2b_a')(conv3d_2b_a)
    conv3d_2b_a = layers.Activation('relu', name='ReLU_2b_a')(conv3d_2b_a)

    conv3d_2b_b = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_2b_b')(conv3d_2b_a)
    conv3d_2b_b = layers.BatchNormalization(name='BatchNorm_2b_b')(conv3d_2b_b)

    conv3d_2b = layers.Add(name='Add_2b')([conv3d_2a, conv3d_2b_b])
    conv3d_2b = layers.Activation('relu', name='ReLU_2b')(conv3d_2b)

    # Res3D Block 3

    # 尺度变换层
    conv3d_3a_1 = layers.Conv3D(128, (1, 1, 1), strides=(2, 2, 2), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_3a_1')(conv3d_2b)
    conv3d_3a_1 = layers.BatchNormalization(name='BatchNorm_3a_1')(conv3d_3a_1)
    conv3d_3a_a = layers.Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_3a_a')(conv3d_2b)
    conv3d_3a_a = layers.BatchNormalization(name='BatchNorm_3a_a')(conv3d_3a_a)
    conv3d_3a_a = layers.Activation('relu', name='ReLU_3a_a')(conv3d_3a_a)
    conv3d_3a_b = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_3a_b')(conv3d_3a_a)
    conv3d_3a_b = layers.BatchNormalization(name='BatchNorm_3a_b')(conv3d_3a_b)
    conv3d_3a = layers.Add(name='Add_3a')([conv3d_3a_1, conv3d_3a_b])
    conv3d_3a = layers.Activation('relu', name='ReLU_3a')(conv3d_3a)

    conv3d_3b_a = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_3b_a')(conv3d_3a)
    conv3d_3b_a = layers.BatchNormalization(name='BatchNorm_3b_a')(conv3d_3b_a)
    conv3d_3b_a = layers.Activation('relu', name='ReLU_3b_a')(conv3d_3b_a)
    conv3d_3b_b = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_3b_b')(conv3d_3b_a)
    conv3d_3b_b = layers.BatchNormalization(name='BatchNorm_3b_b')(conv3d_3b_b)
    conv3d_3b = layers.Add(name='Add_3b')([conv3d_3a, conv3d_3b_b])
    conv3d_3b = layers.Activation('relu', name='ReLU_3b')(conv3d_3b)

    # Res3D Block 4
    conv3d_4a_1 = layers.Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_4a_1')(conv3d_3b)
    conv3d_4a_1 = layers.BatchNormalization(name='BatchNorm_4a_1')(conv3d_4a_1)
    conv3d_4a_a = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_4a_a')(conv3d_3b)
    conv3d_4a_a = layers.BatchNormalization(name='BatchNorm_4a_a')(conv3d_4a_a)
    conv3d_4a_a = layers.Activation('relu', name='ReLU_4a_a')(conv3d_4a_a)
    conv3d_4a_b = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_4a_b')(conv3d_4a_a)
    conv3d_4a_b = layers.BatchNormalization(name='BatchNorm_4a_b')(conv3d_4a_b)
    conv3d_4a = layers.Add(name='Add_4a')([conv3d_4a_1, conv3d_4a_b])
    conv3d_4a = layers.Activation('relu', name='ReLU_4a')(conv3d_4a)

    conv3d_4b_a = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_4b_a')(conv3d_4a)
    conv3d_4b_a = layers.BatchNormalization(name='BatchNorm_4b_a')(conv3d_4b_a)
    conv3d_4b_a = layers.Activation('relu', name='ReLU_4b_a')(conv3d_4b_a)
    conv3d_4b_b = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                dilation_rate=(1, 1, 1), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay), use_bias=False,
                                name='Conv3D_4b_b')(conv3d_4b_a)
    conv3d_4b_b = layers.BatchNormalization(name='BatchNorm_4b_b')(conv3d_4b_b)
    conv3d_4b = layers.Add(name='Add_4b')([conv3d_4a, conv3d_4b_b])
    conv3d_4b = layers.Activation('relu', name='ReLU_4b')(conv3d_4b)

    return conv3d_4b


def relu6(x):
    return layers.ReLU(max_value=6)(x)


def mobilenet(inputs, weight_decay):
    conv2d_1a = layers.SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_1a')(inputs)
    conv2d_1a = layers.BatchNormalization(name='BatchNorm_Conv2d_1a')(conv2d_1a)
    conv2d_1a = layers.Activation(relu6, name='ReLU_Conv2d_1a')(conv2d_1a)

    conv2d_1b = layers.SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_1b')(conv2d_1a)
    conv2d_1b = layers.BatchNormalization(name='BatchNorm_Conv2d_1b')(conv2d_1b)
    conv2d_1b = layers.Activation(relu6, name='ReLU_Conv2d_1b')(conv2d_1b)

    conv2d_2a = layers.SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_2a')(conv2d_1b)
    conv2d_2a = layers.BatchNormalization(name='BatchNorm_Conv2d_2a')(conv2d_2a)
    conv2d_2a = layers.Activation(relu6, name='ReLU_Conv2d_2a')(conv2d_2a)

    conv2d_2b = layers.SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_2b')(conv2d_2a)
    conv2d_2b = layers.BatchNormalization(name='BatchNorm_Conv2d_2b')(conv2d_2b)
    conv2d_2b = layers.Activation(relu6, name='ReLU_Conv2d_2b')(conv2d_2b)

    conv2d_2c = layers.SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_2c')(conv2d_2b)
    conv2d_2c = layers.BatchNormalization(name='BatchNorm_Conv2d_2c')(conv2d_2c)
    conv2d_2c = layers.Activation(relu6, name='ReLU_Conv2d_2c')(conv2d_2c)

    conv2d_2d = layers.SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_2d')(conv2d_2c)
    conv2d_2d = layers.BatchNormalization(name='BatchNorm_Conv2d_2d')(conv2d_2d)
    conv2d_2d = layers.Activation(relu6, name='ReLU_Conv2d_2d')(conv2d_2d)

    conv2d_2e = layers.SeparableConv2D(512, (3, 3), strides=(1, 1), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_2e')(conv2d_2d)
    conv2d_2e = layers.BatchNormalization(name='BatchNorm_Conv2d_2e')(conv2d_2e)
    conv2d_2e = layers.Activation(relu6, name='ReLU_Conv2d_2e')(conv2d_2e)

    conv2d_3a = layers.SeparableConv2D(1024, (3, 3), strides=(2, 2), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_3a')(conv2d_2e)
    conv2d_3a = layers.BatchNormalization(name='BatchNorm_Conv2d_3a')(conv2d_3a)
    conv2d_3a = layers.Activation(relu6, name='ReLU_Conv2d_3a')(conv2d_3a)

    conv2d_3b = layers.SeparableConv2D(1024, (3, 3), strides=(2, 2), padding='same',
                                       depthwise_regularizer=l2(weight_decay),
                                       pointwise_regularizer=l2(weight_decay),
                                       name='SeparableConv2D_3b')(conv2d_3a)
    conv2d_3b = layers.BatchNormalization(name='BatchNorm_Conv2d_3b')(conv2d_3b)
    conv2d_3b = layers.Activation(relu6, name='ReLU_Conv2d_3b')(conv2d_3b)

    return conv2d_3b


def reshape_bz1(x):
    return K.reshape(x, (32, 28, 28, 256))


def reshape_bz2(x):
    return K.reshape(x, (2, 16, 4, 4, 1024))


def res3d_clstm_mobilenet(inputs, seq_len, weight_decay):
    # Res3D Block
    res3d_featmap = res3d(inputs, weight_decay)   # 2x28x28x256

    # GatedConvLSTM2D Block
    clstm2d_1 = layers.GatedConvLSTM2D(256, (3, 3), strides=(1, 1), padding='same',
                                       kernel_initializer='he_normal', recurrent_initializer='he_normal',
                                       kernel_regularizer=l2(weight_decay),
                                       recurrent_regularizer=l2(weight_decay),
                                       return_sequences=True, name='gatedclstm2d_1')(res3d_featmap)
    clstm2d_2 = layers.GatedConvLSTM2D(256, (3, 3), strides=(1, 1), padding='same',
                                       kernel_initializer='he_normal', recurrent_initializer='he_normal',
                                       kernel_regularizer=l2(weight_decay),
                                       recurrent_regularizer=l2(weight_decay),
                                       return_sequences=True, name='gatedclstm2d_2')(clstm2d_1)

    # featmap_2d = layers.Reshape((28, 28, 256), name='clstm_reshape')(clstm2d_2)
    featmap_2d = layers.Lambda(reshape_bz1, name='clstm_reshape')(clstm2d_2)
    print(featmap_2d.get_shape().as_list())
    # MobileNet
    features = mobilenet(featmap_2d, weight_decay)
    print(features.get_shape().as_list())
    # features = layers.Reshape((int(seq_len / 2), 4, 4, 1024), name='feature_reshape')(features)
    features = layers.Lambda(reshape_bz2, name='feature_reshape')(features)
    print(features.get_shape().as_list())
    gpooling = layers.AveragePooling3D(pool_size=(seq_len / 2, 4, 4), strides=(seq_len / 2, 4, 4),
                                       padding='valid', name='Average_Pooling')(features)

    return gpooling





