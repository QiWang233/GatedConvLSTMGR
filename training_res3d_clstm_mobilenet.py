# -*- coding: utf-8 -*-
# @Time : 2022/8/2 10:32
# @File : training_res3d_clstm_mobilenet.py
# @Software: PyCharm
# @Author : @white233

import os
import numpy as np
import tensorflow as tf

import networks.inputs as data
from networks.res3d_clstm_mobilenet import res3d_clstm_mobilenet
from networks.callbacks import LearningRateScheduler
from networks.datagen import DiyGesTrainImageGenerator, DiyGesTestImageGenerator
from tensorflow.python.keras import layers, models, regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import argparse

parser = argparse.ArgumentParser(description='train model way')
parser.add_argument('--way', type=int, default=10, help='value may like head=10,tail=20,head_tail=30,whole_body=40')
parser.add_argument('--gpu', type=str, default='0', help='--gpu \'0\' or \'1\'')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('CUDA_VISIBLE_DEVICES==' + args.gpu)

l2 = regularizers.l2

RGB = 0
Depth = 1
Flow = 2

# Dataset
JESTER = 0
ISOGD = 1
DIYGD = 2

cfg_modality = Depth
cfg_dataset = DIYGD

if cfg_modality == RGB:
    str_modality = 'rgb'
elif cfg_modality == Depth:
    str_modality = 'depth'
elif cfg_modality == Flow:
    str_modality = 'flow'
else:
    str_modality = None

if cfg_dataset == JESTER:
    nb_epoch = 30
    init_epoch = 0
    seq_len = 16
    batch_size = 16
    num_classes = 27
    dataset_name = 'jester_%s' % str_modality
    training_datalist = './dataset_splits/Jester/train_%s_list.txt' % str_modality
    testing_datalist = './dataset_splits/Jester/valid_%s_list.txt' % str_modality
elif cfg_dataset == ISOGD:
    nb_epoch = 10
    init_epoch = 0
    seq_len = 32
    batch_size = 2
    num_classes = 249
    dataset_name = 'isogr_%s' % str_modality
    training_datalist = './dataset_splits/IsoGD/train_%s_list.txt' % str_modality
    testing_datalist = './dataset_splits/IsoGD/valid_%s_list.txt' % str_modality
elif cfg_dataset == DIYGD:
    nb_epoch = 10
    init_epoch = 0
    seq_len = 32
    batch_size = 2
    num_classes = 8
    dataset_name = '%d_diygr_%s' % (args.way, str_modality)

    training_datalist = './dataset_splits/DiyGD/%d_train_%s_list.txt' % (args.way, str_modality)
    testing_datalist = './dataset_splits/DiyGD/%d_valid_%s_list.txt' % (args.way, str_modality)

else:
    nb_epoch = 0
    init_epoch = 0
    seq_len = 0
    batch_size = 0
    num_classes = 0
    dataset_name = None
    training_datalist = None
    testing_datalist = None

weight_decay = 0.00005
model_prefix = './models'
weights_file = '%s/%s_weights.{epoch:02d}-{val_loss:.2f}.h5' % (model_prefix, dataset_name)

_, train_labels = data.load_diy_video_list(training_datalist)
train_steps = len(train_labels) / batch_size
_, test_labels = data.load_diy_video_list(testing_datalist)
test_steps = len(test_labels) / batch_size
print(training_datalist, )
print(
    'nb_epoch: %d - seq_len: %d - batch_size: %d - weight_decay: %.6f' % (nb_epoch, seq_len, batch_size, weight_decay))


def lr_polynomial_decay(global_step):
    learning_rate = 0.001
    end_learning_rate = 0.000001
    decay_steps = train_steps * nb_epoch
    power = 0.9
    p = float(global_step) / float(decay_steps)
    lr = (learning_rate - end_learning_rate) * np.power(1 - p, power) + end_learning_rate
    return lr


inputs = layers.Input(shape=(seq_len, 112, 112, 1), batch_size=batch_size)
feature = res3d_clstm_mobilenet(inputs, seq_len, weight_decay)
flatten = layers.Flatten(name='Flatten')(feature)
classes = layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = layers.Activation('softmax', name='Output')(classes)
model = models.Model(inputs=inputs, outputs=outputs)

# pretrained_model = '%s/isogr_%s_gatedclstm_weights.h5' % (model_prefix, str_modality)
# print('Loading pretrained model from %s' % pretrained_model)
# model.load_weights(pretrained_model, by_name=False)
# for i in range(len(model.trainable_weights)):
#     print(model.trainable_weights[i])

optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_reducer = LearningRateScheduler(lr_polynomial_decay, train_steps)

model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc",
                                   save_best_only=False, save_weights_only=True, mode='auto')
callbacks = [lr_reducer, model_checkpoint]

model.fit(DiyGesTrainImageGenerator(training_datalist, batch_size, seq_len, num_classes, cfg_modality),
          steps_per_epoch=train_steps,
          epochs=nb_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=DiyGesTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes,
                                                   cfg_modality),
          validation_steps=test_steps,
          initial_epoch=init_epoch,
          )
