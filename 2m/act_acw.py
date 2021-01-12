import os
import csv
import datetime as dt
import numpy as np
import sklearn.metrics as metrics
from keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, LSTM, TimeDistributed, Reshape, concatenate
from keras.models import Model
import pandas as pd
import keras.backend as K
import random
from scipy import fftpack
from keras.utils import np_utils
from tensorflow import set_random_seed
import sys

random.seed(0)
np.random.seed(1)
set_random_seed(2)
frame_size = 3*1

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

act_path = '/home/mex/data/act/'
acw_path = '/home/mex/data/acw/'
results_file = '/home/mex/results_lopo/2m/dct_ac_2m_lstm.csv'

frames_per_second = 100
window = 5
increment = 2
dct_length = 60
feature_length = dct_length * 3

ac_min_length = 95*window
ac_max_length = 100*window
fusion = int(sys.argv[1])


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


def split_windows(act_data, acw_data):
    outputs = []
    start = act_data[0][0]
    end = act_data[len(act_data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    act_frames = [a[1:] for a in act_data[:]]
    act_frames = np.array(act_frames)
    act_length = act_frames.shape[0]
    act_frames = np.reshape(act_frames, (act_length*frame_size))
    act_frames = act_frames/(max(act_frames)-min(act_frames))
    act_frames = [float("{0:.5f}".format(f)) for f in act_frames.tolist()]
    act_frames = np.reshape(np.array(act_frames), (act_length, frame_size))

    acw_frames = [a[1:] for a in acw_data[:]]
    acw_frames = np.array(acw_frames)
    acw_length = acw_frames.shape[0]
    acw_frames = np.reshape(acw_frames, (acw_length*frame_size))
    acw_frames = acw_frames/(max(acw_frames)-min(acw_frames))
    acw_frames = [float("{0:.5f}".format(f)) for f in acw_frames.tolist()]
    acw_frames = np.reshape(np.array(acw_frames), (acw_length, frame_size))

    while start + _window < end:
        _end = start + _window
        act_start_index = find_index(act_data, start)
        act_end_index = find_index(act_data, _end)
        acw_start_index = find_index(acw_data, start)
        acw_end_index = find_index(acw_data, _end)
        act_instances = [a[:] for a in act_frames[act_start_index:act_end_index]]
        acw_instances = [a[:] for a in acw_frames[acw_start_index:acw_end_index]]
        start = start + _increment
        instances = [act_instances, acw_instances]
        outputs.append(instances)
    return outputs


# single sensor
def extract_features(act_data, acw_data):
    _features = {}
    for subject in act_data:
        _activities = {}
        act_activities = act_data[subject]
        for act_activity in act_activities:
            time_windows = []
            activity_id = activity_id_dict.get(act_activity)
            act_activity_data = act_activities[act_activity]
            acw_activity_data = acw_data[subject][act_activity]
            for item in act_activity_data.keys():
                time_windows.extend(split_windows(act_activity_data[item], acw_activity_data[item]))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def dct(data):
    new_data = []
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], 2, window, frames_per_second, 3))
    for item in data:
        new_item = []
        for it in item:
            new_its = []
            for i in range(it.shape[0]):
                if dct_length > 0:
                    x = [t[0] for t in it[i]]
                    y = [t[1] for t in it[i]]
                    z = [t[2] for t in it[i]]

                    dct_x = np.abs(fftpack.dct(x, norm='ortho'))
                    dct_y = np.abs(fftpack.dct(y, norm='ortho'))
                    dct_z = np.abs(fftpack.dct(z, norm='ortho'))

                    v = np.array([])
                    v = np.concatenate((v, dct_x[:dct_length]))
                    v = np.concatenate((v, dct_y[:dct_length]))
                    v = np.concatenate((v, dct_z[:dct_length]))
                    new_its.append(v)
            new_item.append(new_its)
        new_data.append(new_item)
    return new_data


def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return dct(flatten_data), flatten_labels


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
        new_activities = {}
        activities = _features[subject]
        for act in activities:
            items = activities[act]
            new_items = []
            for item in items:
                new_item = []
                act_len = len(item[0])
                acw_len = len(item[1])
                if act_len < ac_min_length or acw_len < ac_min_length:
                    continue
                if act_len > ac_max_length:
                    new_item.append(reduce(item[0], act_len - ac_max_length))
                elif act_len < ac_max_length:
                    new_item.append(pad(item[0], ac_max_length - act_len))
                else:
                    new_item.append(item[0])

                if acw_len > ac_max_length:
                    new_item.append(reduce(item[1], acw_len - ac_max_length))
                elif acw_len < ac_max_length:
                    new_item.append(pad(item[1], ac_max_length - acw_len))
                else:
                    new_item.append(item[1])
                new_items.append(new_item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


def build_late_fusion():
    input_t = Input(shape=(window, feature_length, 1))
    input_w = Input(shape=(window, feature_length, 1))

    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_t)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]*K.int_shape(x)[3]))(x)
    x = LSTM(1200)(x)
    x = BatchNormalization()(x)
    x = Dense(600, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)

    y = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_w)
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

    model = Model(inputs=[input_t, input_w], outputs=z)
    model.summary()
    return model


def build_early_fusion():
    input_t = Input(shape=(window, feature_length, 1))
    input_w = Input(shape=(window, feature_length, 1))

    input_tw = concatenate([input_t, input_w], axis=2)

    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_tw)
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

    model = Model(inputs=[input_t, input_w], outputs=x)
    model.summary()
    return model


def build_mid_fusion():
    input_t = Input(shape=(window, feature_length, 1))
    input_w = Input(shape=(window, feature_length, 1))

    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_t)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]*K.int_shape(x)[3]))(x)
    x = LSTM(1200)(x)
    x = BatchNormalization()(x)

    y = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_w)
    y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Conv1D(64, kernel_size=5, activation='relu'))(y)
    y = TimeDistributed(MaxPooling1D(pool_size=2))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = Reshape((K.int_shape(y)[1], K.int_shape(y)[2]*K.int_shape(y)[3]))(y)
    y = LSTM(1200)(y)
    y = BatchNormalization()(y)

    z = concatenate([x, y])
    z = Dense(600, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dense(100, activation='relu')(z)
    z = BatchNormalization()(z)

    z = Dense(len(activity_list), activation='softmax')(z)

    model = Model(inputs=[input_t, input_w], outputs=z)
    model.summary()
    return model


def _run_(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    _train_features = np.expand_dims(_train_features, 5)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.expand_dims(_test_features, 5)
    print(_test_features.shape)

    _train_features_t = _train_features[:, 0]
    _train_features_w = _train_features[:, 1]
    print(_train_features_t.shape)
    print(_train_features_w.shape)

    _test_features_t = _test_features[:, 0]
    _test_features_w = _test_features[:, 1]
    print(_test_features_t.shape)
    print(_test_features_w.shape)

    if fusion == 0:
        model = build_early_fusion()
    elif fusion == 1:
        model = build_mid_fusion()
    else:
        model = build_late_fusion()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([_train_features_t, _train_features_w], _train_labels, verbose=1, batch_size=64, epochs=30, shuffle=True)
    _predict_labels = model.predict([_test_features_t, _test_features_w], batch_size=64, verbose=0)
    f_score = metrics.f1_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(_test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'dct_ac_2m_lstm' + ',' + str(fusion) + ',' + str(sys.argv[2]) + ',' + str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, str(results))

    # _test_labels = pd.Series(_test_labels.argmax(axis=1), name='Actual')
    # _predict_labels = pd.Series(_predict_labels.argmax(axis=1), name='Predicted')
    # df_confusion = pd.crosstab(_test_labels, _predict_labels)
    # print(df_confusion)
    # write_data(results_file, str(df_confusion))


act_data = read(act_path, '_act')
acw_data = read(acw_path, '_acw')

all_features = extract_features(act_data, acw_data)

all_features = pad_features(all_features)
all_features = frame_reduce(all_features)
all_users = list(all_features.keys())

i = sys.argv[2]
train_features, test_features = train_test_split(all_features, [i])

train_features, train_labels = flatten(train_features)
test_features, test_labels = flatten(test_features)

train_labels = np_utils.to_categorical(train_labels, len(activity_list))
test_labels = np_utils.to_categorical(test_labels, len(activity_list))

_run_(train_features, train_labels, test_features, test_labels)
