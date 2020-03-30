from __future__ import print_function

import numpy as np
import keras
import os
import json
import tqdm

from TF.code_.ecg_master.ecg import load,util
from sklearn.metrics import confusion_matrix, classification_report

def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    model = keras.models.load_model(model_path)

    # 将data_json分成多段分别预测结果，最后组合混淆矩阵
    Y, Y_probs = [], []
    batch_size = 32
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    num_examples = len(data)
    end = num_examples - batch_size + 1
    batches = [data[i:i + batch_size]
               for i in range(0, end, batch_size)]
    for batch in tqdm.tqdm(batches):
        labels = []
        eegs = []
        for d in batch:
            labels.append(d['labels'])
            eegs.append(load.load_ecg(d['ecg']))
        test_x,test_y = preproc.process(eegs, labels)
        probs = model.predict(test_x, verbose=0)
        if len(Y):
            Y = np.concatenate([Y,test_y],axis = 0)
            Y_probs = np.concatenate([Y_probs, probs], axis = 0)
        else:
            Y, Y_probs = test_y, probs

    # gen_x, gen_y = load.data_generator_eeg(batch_size, preproc, data_json)
    # probs = keras.Model.predict_generator(gen_x,verbose=1)


    #评估结果
    y_ = np.reshape(np.argmax(Y,axis=2),[-1])
    pre_ = np.reshape(np.argmax(Y_probs, axis=2),[-1])

    #每个类的各项指标
    cm = confusion_matrix(y_, pre_)
    np.set_printoptions(precision=3)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    print(classification_report(y_, pre_))

    from use_ import confusion_matrix_plot_
    confusion_matrix_plot_.plot_cm(['move','stage 1','stage 2','stage 3','stage 4','R','W'], y_, pre_)

    return Y_probs

if __name__ == '__main__':
    probs = predict('E:\PythonWorkSpace\pycharm\TF\code_\sleep_stage\eeg\dev.json',
                    'E:\PythonWorkSpace\pycharm\TF\code_\sleep_stage\eeg\saved\experiment\\1576916981-234\\0.185-0.921-011-0.160-0.938.hdf5')


