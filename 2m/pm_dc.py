import os
import csv
import datetime as dt
import numpy as np
import sklearn.metrics as metrics
from keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, LSTM, TimeDistributed, Reshape, concatenate, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import keras.backend as K
import random
from scipy import fftpack
from keras.utils import np_utils
from tensorflow import set_random_seed
import sys

random.seed(0)
np.random.seed(1)

pm_frame_size = 16*16
dc_frame_size = 16*12

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

pm_path = '/home/mex/data/pm_scaled/1.0_0.5'
dc_path = '/home/mex/data/dc_scaled/0.05_0.05'

results_file = '/home/mex/results_lopo/2m/pm_dc_2m_lstm_2dcnn.csv'

frames_per_second = 1

window = 5
increment = 2
fusion = int(sys.argv[1])

min_length = frames_per_second*window
max_length = 15*window


def write_data(file_path, data):
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(data + '\n')
    else:
        f = open(file_path, 'w')
        f.write(data + '\n')
    f.close()


def _read(_file):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row[0]) == 19 and '.' not in row[0]:
            row[0] = row[0]+'.000000'
        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
        _temp = [float(f) for f in row[1:]]
        temp.extend(_temp)
        _data.append(temp)
    return _data


def read(path, _sensor):
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        activities = [f for f in activities if not f.startswith('.')]
        for activity in activities:
            sensor = activity.split('.')[0].replace(_sensor, '')
            activity_id = sensor.split('_')[0]
            sensor_index = sensor.split('_')[1]
            _data = _read(os.path.join(subject_path, activity), )
            if activity_id in allactivities:
                allactivities[activity_id][sensor_index] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor_index] = _data
        alldata[subject] = allactivities
    return alldata


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim(_data):
    _length = len(_data)
    _inc = _length/(window*frames_per_second)
    _new_data = []
    for i in range(window*frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def frame_reduce(_features):
    if frames_per_second == 0:
        return _features
    new_features = {}
    for subject in _features:
        _activities = {}
        activities = _features[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                new_item = []
                new_item.append(trim(item[0]))
                new_item.append(trim(item[1]))
                time_windows.append(new_item)
            _activities[activity] = time_windows
        new_features[subject] = _activities
    return new_features


def split_windows(pm_data, dc_data):
    outputs = []
    start = max(pm_data[0][0], dc_data[0][0])
    end = min(pm_data[len(pm_data) - 1][0], dc_data[len(dc_data) - 1][0])
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    pm_frames = [a[1:] for a in pm_data[:]]
    dc_frames = [a[1:] for a in dc_data[:]]

    pm_frames = np.array(pm_frames)
    pm_length = pm_frames.shape[0]
    pm_frames = np.reshape(pm_frames, (pm_length*pm_frame_size))
    pm_frames = pm_frames/(max(pm_frames)-min(pm_frames))
    pm_frames = [float("{0:.5f}".format(f)) for f in pm_frames.tolist()]
    pm_frames = np.reshape(np.array(pm_frames), (pm_length, pm_frame_size))

    dc_frames = np.array(dc_frames)

    while start + _window < end:
        _end = start + _window
        pm_start_index = find_index(pm_data, start)
        pm_end_index = find_index(pm_data, _end)
        dc_start_index = find_index(dc_data, start)
        dc_end_index = find_index(dc_data, _end)
        pm_instances = [a[:] for a in pm_frames[pm_start_index:pm_end_index]]
        dc_instances = [a[:] for a in dc_frames[dc_start_index:dc_end_index]]
        start = start + _increment
        instances = [pm_instances, dc_instances]
        outputs.append(instances)
    return outputs


def extract_features(pm_data, dc_data):
    _features = {}
    for subject in pm_data:
        _pmivities = {}
        pm_pmivities = pm_data[subject]
        for pm_pmivity in pm_pmivities:
            time_windows = []
            pmivity_id = activity_id_dict.get(pm_pmivity)
            pm_pmivity_data = pm_data[subject][pm_pmivity]
            dc_pmivity_data = dc_data[subject][pm_pmivity]
            for item in pm_pmivity_data.keys():
                time_windows.extend(split_windows(pm_pmivity_data[item], dc_pmivity_data[item]))
            _pmivities[pmivity_id] = time_windows
        _features[subject] = _pmivities
    return _features


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def flatten(_data):
    flatten_data = []
    flatten_labels = []
    for subject in _data:
        pmivities = _data[subject]
        for pmivity in pmivities:
            pmivity_data = pmivities[pmivity]
            flatten_data.extend(pmivity_data)
            flatten_labels.extend([pmivity for i in range(len(pmivity_data))])

    pm = [f[0] for f in flatten_data]
    dc = [f[1] for f in flatten_data]
    return pm, dc, flatten_labels


def pad(data, length):
    pad_length = []
    if length % 2 == 0:
        pad_length = [int(length / 2), int(length / 2)]
    else:
        pad_length = [int(length / 2) + 1, int(length / 2)]
    new_data = []
    for index in range(pad_length[0]):
        new_data.append(data[0])
    new_data.extend(data)
    for index in range(pad_length[1]):
        new_data.append(data[len(data) - 1])
    return new_data


def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data


def pad_features(_features):
    new_features = {}
    for subject in _features:
        new_pmivities = {}
        pmivities = _features[subject]
        for pm in pmivities:
            items = pmivities[pm]
            new_items = []
            for item in items:
                new_item = []
                pm_len = len(item[0])
                dc_len = len(item[1])
                if dc_len < min_length:
                    continue

                if pm_len > max_length:
                    new_item.append(reduce(item[0], pm_len - max_length))
                elif pm_len < max_length:
                    new_item.append(pad(item[0], max_length - pm_len))
                else:
                    new_item.append(item[0])

                if dc_len > max_length:
                    new_item.append(reduce(item[1], dc_len - max_length))
                elif dc_len < max_length:
                    new_item.append(pad(item[1], max_length - dc_len))
                else:
                    new_item.append(item[1])

                new_items.append(new_item)
            new_pmivities[pm] = new_items
        new_features[subject] = new_pmivities
    return new_features


def build_early_fusion():
    input_dc = Input(shape=(window, frames_per_second*dc_frame_size, 1))
    input_pm = Input(shape=(window, frames_per_second*pm_frame_size, 1))

    input_pmdc = concatenate([input_pm, input_dc], axis=2)

    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_pmdc)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(128, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(256, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]*K.int_shape(x)[3]))(x)
    x = LSTM(1200)(x)
    x = BatchNormalization()(x)
    x = Dense(600, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(activity_list), activation='softmax')(x)

    model = Model(inputs=[input_pm, input_dc], outputs=x)
    model.summary()
    return model


def build_mid_fusion():
    input_dc = Input(shape=(12, 16 * window * frames_per_second, 1))
    input_pm = Input(shape=(window, frames_per_second*pm_frame_size, 1))

    x = Conv2D(32, kernel_size=(3,3), activation='relu')(input_dc)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)

    y = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_pm)
    y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(y)
    y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = Reshape((K.int_shape(y)[1], K.int_shape(y)[2]*K.int_shape(y)[3]))(y)
    y = LSTM(1200)(y)
    y = BatchNormalization()(y)

    z = concatenate([x, y])
    z = Dense(640, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dense(240, activation='relu')(z)
    z = BatchNormalization()(z)

    z = Dense(len(activity_list), activation='softmax')(z)

    model = Model(inputs=[input_pm, input_dc], outputs=z)
    model.summary()
    return model


def build_late_fusion():
    input_dc = Input(shape=(12, 16 * window * frames_per_second, 1))
    input_pm = Input(shape=(window, frames_per_second*pm_frame_size, 1))

    x = Conv2D(32, kernel_size=(3,3), activation='relu')(input_dc)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(600, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)

    y = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_pm)
    y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(y)
    y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = Reshape((K.int_shape(y)[1], K.int_shape(y)[2]*K.int_shape(y)[3]))(y)
    y = LSTM(1200)(y)
    y = BatchNormalization()(y)
    y = Dense(600, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dense(100, activation='relu')(y)
    y = BatchNormalization()(y)

    z = concatenate([x, y])
    z = Dense(len(activity_list), activation='softmax')(z)

    model = Model(inputs=[input_pm, input_dc], outputs=z)
    model.summary()
    return model


def _run_(pm_train_features, dc_train_features, train_labels, pm_test_features, dc_test_features, test_labels):
    dc_train_features = np.array(dc_train_features)

    dc_test_features = np.array(dc_test_features)

    pm_train_features = np.array(pm_train_features)
    pm_train_features = np.reshape(pm_train_features, (pm_train_features.shape[0], window, frames_per_second*16*16))
    pm_train_features = np.expand_dims(pm_train_features, 4)
    print(pm_train_features.shape)

    pm_test_features = np.array(pm_test_features)
    pm_test_features = np.reshape(pm_test_features, (pm_test_features.shape[0], window, frames_per_second*16*16))
    pm_test_features = np.expand_dims(pm_test_features, 4)
    print(pm_test_features.shape)

    if fusion == 0:
        dc_train_features = np.reshape(dc_train_features, (dc_train_features.shape[0], window, frames_per_second*16*12))
        dc_train_features = np.expand_dims(dc_train_features, 4)
        print(dc_train_features.shape)

        dc_test_features = np.reshape(dc_test_features, (dc_test_features.shape[0], window, frames_per_second*16*12))
        dc_test_features = np.expand_dims(dc_test_features, 4)
        print(dc_test_features.shape)

        model = build_early_fusion()
    else:
        dc_train_features = np.reshape(dc_train_features, (dc_train_features.shape[0], dc_train_features.shape[1], 12, 16))
        dc_train_features = np.swapaxes(dc_train_features, 1, 2)
        dc_train_features = np.swapaxes(dc_train_features, 2, 3)
        dc_train_features = np.reshape(dc_train_features, (dc_train_features.shape[0], dc_train_features.shape[1],
                                                           dc_train_features.shape[2] * dc_train_features.shape[3]))
        dc_train_features = np.expand_dims(dc_train_features, 4)
        print(dc_train_features.shape)

        dc_test_features = np.reshape(dc_test_features, (dc_test_features.shape[0], dc_test_features.shape[1], 12, 16))
        dc_test_features = np.swapaxes(dc_test_features, 1, 2)
        dc_test_features = np.swapaxes(dc_test_features, 2, 3)
        dc_test_features = np.reshape(dc_test_features, (dc_test_features.shape[0], dc_test_features.shape[1],
                                                         dc_test_features.shape[2] * dc_test_features.shape[3]))
        dc_test_features = np.expand_dims(dc_test_features, 4)
        print(dc_test_features.shape)

        if fusion == 1:
            model = build_mid_fusion()
        else:
            model = build_late_fusion()

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([pm_train_features, dc_train_features], train_labels, verbose=1, batch_size=32, epochs=30, shuffle=True)
    _predict_labels = model.predict([pm_test_features, dc_test_features], batch_size=64, verbose=0)
    f_score = metrics.f1_score(test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'pm_dc' + ',' + str(fusion) + ',' + str(sys.argv[2]) + ',' + str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, str(results))


_pm_data = read(pm_path, '_pm')
_dc_data = read(dc_path, '_dc')

all_features = extract_features(_pm_data, _dc_data)

all_features = pad_features(all_features)
all_features = frame_reduce(all_features)

i = sys.argv[2]
set_random_seed(2)

train_features, test_features = train_test_split(all_features, [i])

_pm_train_features, _dc_train_features, _train_labels = flatten(train_features)
_pm_test_features, _dc_test_features, _test_labels = flatten(test_features)

_train_labels = np_utils.to_categorical(_train_labels, len(activity_list))
_test_labels = np_utils.to_categorical(_test_labels, len(activity_list))

_run_(_pm_train_features, _dc_train_features, _train_labels, _pm_test_features, _dc_test_features, _test_labels)
