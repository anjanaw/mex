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

ac_frame_size = 3*1
dc_frame_size = 16*12

acwivity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(acwivity_list))
acwivity_id_dict = dict(zip(acwivity_list, id_list))

acw_path = '/home/mex/data/acw/'
dc_path = '/home/mex/data/dc_scaled/0.05_0.05'

results_file = '/home/mex/results_lopo/2m/dc_acw_2m_2dcnn_lstm.csv'

dc_frames_per_second = 1
ac_frames_per_second = 100

window = 5
increment = 2
dct_length = 60
feature_length = dct_length * 3
fusion = int(sys.argv[1])

ac_min_length = 95*window
ac_max_length = 100*window
dc_min_length = dc_frames_per_second*window
dc_max_length = 15*window


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
        allacwivities = {}
        subject_path = os.path.join(path, subject)
        acwivities = os.listdir(subject_path)
        acwivities = [f for f in acwivities if not f.startswith('.')]
        for acwivity in acwivities:
            sensor = acwivity.split('.')[0].replace(_sensor, '')
            acwivity_id = sensor.split('_')[0]
            sensor_index = sensor.split('_')[1]
            _data = _read(os.path.join(subject_path, acwivity), )
            if acwivity_id in allacwivities:
                allacwivities[acwivity_id][sensor_index] = _data
            else:
                allacwivities[acwivity_id] = {}
                allacwivities[acwivity_id][sensor_index] = _data
        alldata[subject] = allacwivities
    return alldata


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim_ac(_data):
    _length = len(_data)
    _inc = _length/(window*ac_frames_per_second)
    _new_data = []
    for i in range(window*ac_frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def trim_dc(_data):
    _length = len(_data)
    _inc = _length/(window*dc_frames_per_second)
    _new_data = []
    for i in range(window*dc_frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def frame_reduce(_features):
    if dc_frames_per_second == 0:
        return _features
    new_features = {}
    for subject in _features:
        _acwivities = {}
        acwivities = _features[subject]
        for acwivity in acwivities:
            acwivity_data = acwivities[acwivity]
            time_windows = []
            for item in acwivity_data:
                new_item = []
                new_item.append(trim_ac(item[0]))
                new_item.append(trim_dc(item[1]))
                time_windows.append(new_item)
            _acwivities[acwivity] = time_windows
        new_features[subject] = _acwivities
    return new_features


def split_windows(acw_data, dc_data):
    outputs = []
    start = max(acw_data[0][0], dc_data[0][0])
    end = min(acw_data[len(acw_data) - 1][0], dc_data[len(dc_data) - 1][0])
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    acw_frames = [a[1:] for a in acw_data[:]]
    dc_frames = [a[1:] for a in dc_data[:]]

    acw_frames = np.array(acw_frames)
    acw_frames = np.array(acw_frames)
    acw_length = acw_frames.shape[0]
    acw_frames = np.reshape(acw_frames, (acw_length*ac_frame_size))
    acw_frames = acw_frames/(max(acw_frames)-min(acw_frames))
    acw_frames = [float("{0:.5f}".format(f)) for f in acw_frames.tolist()]
    acw_frames = np.reshape(np.array(acw_frames), (acw_length, ac_frame_size))

    dc_frames = np.array(dc_frames)

    while start + _window < end:
        _end = start + _window
        acw_start_index = find_index(acw_data, start)
        acw_end_index = find_index(acw_data, _end)
        dc_start_index = find_index(dc_data, start)
        dc_end_index = find_index(dc_data, _end)
        acw_instances = [a[:] for a in acw_frames[acw_start_index:acw_end_index]]
        dc_instances = [a[:] for a in dc_frames[dc_start_index:dc_end_index]]
        start = start + _increment
        instances = [acw_instances, dc_instances]
        outputs.append(instances)
    return outputs


def extracw_features(acw_data, dc_data):
    _features = {}
    for subject in acw_data:
        _acwivities = {}
        acw_acwivities = acw_data[subject]
        for acw_acwivity in acw_acwivities:
            time_windows = []
            acwivity_id = acwivity_id_dict.get(acw_acwivity)
            acw_acwivity_data = acw_data[subject][acw_acwivity]
            dc_acwivity_data = dc_data[subject][acw_acwivity]
            for item in acw_acwivity_data.keys():
                time_windows.extend(split_windows(acw_acwivity_data[item], dc_acwivity_data[item]))
            _acwivities[acwivity_id] = time_windows
        _features[subject] = _acwivities
    return _features


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def dct(data):
    new_data = []
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], window, ac_frames_per_second, 3))
    for item in data:
        new_item = []
        for i in range(item.shape[0]):
            if dct_length > 0:
                x = [t[0] for t in item[i]]
                y = [t[1] for t in item[i]]
                z = [t[2] for t in item[i]]

                dct_x = np.abs(fftpack.dct(x, norm='ortho'))
                dct_y = np.abs(fftpack.dct(y, norm='ortho'))
                dct_z = np.abs(fftpack.dct(z, norm='ortho'))

                v = np.array([])
                v = np.concatenate((v, dct_x[:dct_length]))
                v = np.concatenate((v, dct_y[:dct_length]))
                v = np.concatenate((v, dct_z[:dct_length]))
                new_item.append(v)
        new_data.append(new_item)
    return new_data


def flatten(_data):
    flatten_data = []
    flatten_labels = []
    for subject in _data:
        acwivities = _data[subject]
        for acwivity in acwivities:
            acwivity_data = acwivities[acwivity]
            flatten_data.extend(acwivity_data)
            flatten_labels.extend([acwivity for i in range(len(acwivity_data))])

    dct_acw = dct([f[0] for f in flatten_data])
    dct_acw = np.array(dct_acw)
    dc = [f[1] for f in flatten_data]
    return dct_acw, dc, flatten_labels


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
        new_acwivities = {}
        acwivities = _features[subject]
        for acw in acwivities:
            items = acwivities[acw]
            new_items = []
            for item in items:
                new_item = []
                acw_len = len(item[0])
                dc_len = len(item[1])
                if dc_len < dc_min_length:
                    continue

                if acw_len > ac_max_length:
                    new_item.append(reduce(item[0], acw_len - ac_max_length))
                elif acw_len < ac_max_length:
                    new_item.append(pad(item[0], ac_max_length - acw_len))
                else:
                    new_item.append(item[0])

                if dc_len > dc_max_length:
                    new_item.append(reduce(item[1], dc_len - dc_max_length))
                elif dc_len < dc_max_length:
                    new_item.append(pad(item[1], dc_max_length - dc_len))
                else:
                    new_item.append(item[1])

                new_items.append(new_item)
            new_acwivities[acw] = new_items
        new_features[subject] = new_acwivities
    return new_features


def build_early_fusion():
    input_dc = Input(shape=(window, dc_frames_per_second*dc_frame_size, 1))
    input_t = Input(shape=(window, feature_length, 1))

    input_tdc = concatenate([input_t, input_dc], axis=2)

    x = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_tdc)
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
    x = Dense(len(acwivity_list), activation='softmax')(x)

    model = Model(inputs=[input_t, input_dc], outputs=x)
    model.summary()
    return model


def build_mid_fusion():
    input_dc = Input(shape=(12, 16 * window * dc_frames_per_second, 1))
    input_t = Input(shape=(window, feature_length, 1))

    x = Conv2D(32, kernel_size=(3,3), activation='relu')(input_dc)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)

    y = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_t)
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
    z = Dense(200, activation='relu')(z)
    z = BatchNormalization()(z)

    z = Dense(len(acwivity_list), activation='softmax')(z)

    model = Model(inputs=[input_t, input_dc], outputs=z)
    model.summary()
    return model


def build_late_fusion():
    input_dc = Input(shape=(12, 16 * window * dc_frames_per_second, 1))
    input_t = Input(shape=(window, feature_length, 1))

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

    y = TimeDistributed(Conv1D(32, kernel_size=5, activation='relu'))(input_t)
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
    z = Dense(len(acwivity_list), activation='softmax')(z)

    model = Model(inputs=[input_t, input_dc], outputs=z)
    model.summary()
    return model


def _run_(acw_train_features, dc_train_features, train_labels, acw_test_features, dc_test_features, test_labels):
    dc_train_features = np.array(dc_train_features)

    dc_test_features = np.array(dc_test_features)

    acw_train_features = np.array(acw_train_features)
    acw_train_features = np.expand_dims(acw_train_features, 3)
    print(acw_train_features.shape)

    acw_test_features = np.array(acw_test_features)
    acw_test_features = np.expand_dims(acw_test_features, 3)
    print(acw_test_features.shape)

    if fusion == 0:
        dc_train_features = np.reshape(dc_train_features, (dc_train_features.shape[0], window, dc_frames_per_second*16*12))
        dc_train_features = np.expand_dims(dc_train_features, 4)
        print(dc_train_features.shape)

        dc_test_features = np.reshape(dc_test_features, (dc_test_features.shape[0], window, dc_frames_per_second*16*12))
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
    model.fit([acw_train_features, dc_train_features], train_labels, verbose=1, batch_size=32, epochs=30, shuffle=True)
    _predict_labels = model.predict([acw_test_features, dc_test_features], batch_size=64, verbose=0)
    f_score = metrics.f1_score(test_labels.argmax(axis=1), _predict_labels.argmax(axis=1), average='macro')
    accuracy = metrics.accuracy_score(test_labels.argmax(axis=1), _predict_labels.argmax(axis=1))
    results = 'acw_dc' + ',' + str(fusion)+',' + str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, str(results))


acw_data = read(acw_path, '_acw')
dc_data = read(dc_path, '_dc')

all_features = extracw_features(acw_data, dc_data)

all_features = pad_features(all_features)
all_features = frame_reduce(all_features)

i = sys.argv[2]
set_random_seed(2)

train_features, test_features = train_test_split(all_features, [i])

_acw_train_features, _dc_train_features, _train_labels = flatten(train_features)
_acw_test_features, _dc_test_features, _test_labels = flatten(test_features)

_train_labels = np_utils.to_categorical(_train_labels, len(acwivity_list))
_test_labels = np_utils.to_categorical(_test_labels, len(acwivity_list))

_run_(_acw_train_features, _dc_train_features, _train_labels, _acw_test_features, _dc_test_features, _test_labels)
