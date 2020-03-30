from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import os
import random
import time
from TF.code_.sleep_stage_beijing import network_sleep_stage,load_data,util

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def train(experiment, params):
    # 统计数据量，数据量过大，不能一次性导入，在训练的时候迭代load
    #计算训练集数据长度
    with open(params['train'], 'r') as fid:
        json_train = [json.loads(l) for l in fid]
    len_train = len(json_train)
    # 计算校验集数据长度
    with open(params['dev'], 'r') as fid:
        json_dev = [json.loads(l) for l in fid]
    len_dev = len(json_dev)
    print("Training size: " + str(len_train) + " examples.")
    print("Dev size: " + str(len_dev) + " examples.")


    #标准化预处理数据，并保存相应均值和方差
    preproc = load_data.Preproc_eeg(params['train'])
    # preproc = util.load(os.path.dirname(model_path))#preproc 与 model 在一个文件夹
    save_dir = make_save_dir(params['save_dir'], experiment)
    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    # 配置网络参数，建立模型
    model = network_sleep_stage.build_network(**params)
    stopping = keras.callbacks.EarlyStopping(patience=8)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=params["learning_rate"] * 0.001)
    checkpointer = keras.callbacks.ModelCheckpoint( filepath=get_filename_for_saving(save_dir), save_best_only=False)
    batch_size = params.get("batch_size", 64)

    # 训练并保存模型
    if params.get("generator", False):
        train_gen = load_data.data_generator_eeg(batch_size, preproc, params['train'])
        dev_gen = load_data.data_generator_eeg(batch_size, preproc, params['dev'])
        #模型将原信号长度通过8次缩放缩小256倍，形成每秒（原长度256）一个标签，同时一秒一个标签
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len_train / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len_dev / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])

if __name__ == '__main__':
    config_file = 'E:\PythonWorkSpace\pycharm\TF\code_\sleep_stage\eeg\config.json'
    experiment = 'experiment'
    params = json.load(open(config_file, 'r'))
    train(experiment, params)
