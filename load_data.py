from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

def data_generator_eeg(batch_size, preproc, data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    num_examples = len(data)
    random.shuffle(data)
    end = num_examples - batch_size + 1
    batches = [data[i:i + batch_size]
               for i in range(0, end, batch_size)]
    # random.shuffle(batches)
    while True:
        for batch in batches:
            labels = []
            eegs = []
            for d in batch:
                labels.append(d['labels'])
                eegs.append(load_ecg(d['ecg']))
            yield preproc.process(eegs, labels)
def data_generator_eog_3000(batch_size, preproc, data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    num_examples = len(data)
    random.shuffle(data)
    end = num_examples - batch_size + 1
    batches = [data[i:i + batch_size]
               for i in range(0, end, batch_size)]
    # random.shuffle(batches)
    while True:
        for batch in batches:
            labels = []
            eegs = []
            for d in batch:
                labels.append(d['labels'])
                eegs.append(load_ecg(d['ecg']))
            yield preproc.process(eegs, labels)
class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return y
class Preproc_eeg:

    def __init__(self, data_json):

        with open(data_json, 'r') as fid:
            data = [json.loads(l) for l in fid]

        #逐个下载并计算均值与std所需条件
        labels = []
        ecgs_sum = 0
        ecg_quare_sum = 0
        lens = 0
        for d in tqdm.tqdm(data):
            labels.append(d['labels'])
            eeg = load_ecg(d['ecg'])
            ecgs_sum = ecgs_sum + sum(eeg)
            lens = lens + np.shape(eeg)[0]
        self.mean = ecgs_sum / lens

        for d in tqdm.tqdm(data):
            eeg = load_ecg(d['ecg'])
            ecg_quare_sum = ecg_quare_sum + sum((eeg-self.mean)**2)
        self.std =  np.sqrt(ecg_quare_sum/lens)

        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}



    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return y



def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_dataset_eeg(data):
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    elif os.path.splitext(record)[1] == ".hdf5":
        import h5py
        with h5py.File(record, 'r') as f:
            # ecg = np.array(f.get('EEG1'))
            # ecg = np.array(f.get('eog'))
            ecg = np.array(f.get(list(f.keys())[0]))
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

if __name__ == "__main__":
    data_json = "examples/cinc17/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break
