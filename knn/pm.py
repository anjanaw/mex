import os
import csv
import datetime as dt
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from keras.layers import Input, Dense
from keras.models import  Model
import os
import random
from sklearn.neighbors import KNeighborsClassifier
import sys

random.seed(0)
np.random.seed(1)

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))

path = '/home/mex/data/pm_scaled/1.0_1.0/'
results_file = '/home/mex/results_lopo/ae_knn_pm.csv'

frames_per_second = 1
window = 5
increment = 2
k = int(sys.argv[1])

pm_min_length = frames_per_second*window
pm_max_length = 15*window


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


def read():
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].replace('_pm', '')
            activity_id = sensor.split('_')[0]
            _data = _read(os.path.join(subject_path, activity), )
            if activity_id in allactivities:
                allactivities[activity_id][sensor] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor] = _data
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


def frame_reduce(_data):
    if frames_per_second == 0:
        return _data
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                time_windows.append(trim(item))
            _activities[activity] = time_windows
        _features[subject] = _activities
    return _features


def split_windows(data):
    outputs = []
    start = data[0][0]
    end = data[len(data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    frames = [a[1:] for a in data[:]]
    frames = np.array(frames)

    while start + _window < end:
        _end = start + _window
        start_index = find_index(data, start)
        end_index = find_index(data, _end)
        instances = [a[:] for a in frames[start_index:end_index]]
        start = start + _increment
        outputs.append(instances)
    return outputs


# single sensor
def extract_features(_data):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            time_windows = []
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            for sensor in activity_data:
                time_windows.extend(split_windows(activity_data[sensor]))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def split(_data, _labels, test_indices):
    _train_data = []
    _train_labels = []
    _test_data = []
    _test_labels = []
    index = 0
    for _datum, _label in zip(_data, _labels):
        if index in test_indices:
            _test_data.append(_datum)
            _test_labels.append(_label)
        else:
            _train_data.append(_datum)
            _train_labels.append(_label)
        index += 1
    return _train_data, _train_labels, _test_data, _test_labels


def train_test_split(user_data, test_ids):
    train_data = {key: value for key, value in user_data.items() if key not in test_ids}
    test_data = {key: value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data


def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return flatten_data, flatten_labels


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
                _len = len(item)
                if _len < pm_min_length:
                    continue
                elif _len > pm_max_length:
                    item = reduce(item, _len - pm_max_length)
                    new_items.append(item)
                elif _len < pm_max_length:
                    item = pad(item, pm_max_length - _len)
                    new_items.append(item)
                elif _len == pm_max_length:
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


def run_knn(_train_features, _train_labels, _test_features, _test_labels):
    _train_features = np.array(_train_features)
    print(_train_features.shape)
    _train_features = np.reshape(_train_features, (_train_features.shape[0],
                                                   _train_features.shape[1]*_train_features.shape[2]))
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    print(_test_features.shape)
    _test_features = np.reshape(_test_features, (_test_features.shape[0],
                                                 _test_features.shape[1]*_test_features.shape[2]))
    print(_test_features.shape)

    # encoded feature representation
    # start
    input_img = Input(shape=(_train_features.shape[1],))
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)

    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(_train_features.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(_train_features, _train_features, epochs=50, batch_size=64, shuffle=True, verbose=0)

    _train_features = encoder.predict(_train_features)
    _test_features = encoder.predict(_test_features)

    print(_train_features.shape)
    print(_test_features.shape)

    # end

    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(_train_features, _train_labels)
    _predict_labels = model.predict(_test_features)
    f_score = metrics.f1_score(_test_labels, _predict_labels, average='macro')
    accuracy = metrics.accuracy_score(_test_labels, _predict_labels)
    results = 'pm,' + str(k) + ',' + str(accuracy)+',' + str(f_score)
    print(results)
    write_data(results_file, results)

    _test_labels = pd.Series(_test_labels, name='Actual')
    _predict_labels = pd.Series(_predict_labels, name='Predicted')
    df_confusion = pd.crosstab(_test_labels, _predict_labels)
    print(df_confusion)
    write_data(results_file, str(df_confusion))


all_data = read()
all_features = extract_features(all_data)
all_data = None
all_features = pad_features(all_features)
all_features = frame_reduce(all_features)
all_users = list(all_features.keys())

train_features, test_features = train_test_split(all_features, [i])

train_features, train_labels = flatten(train_features)
test_features, test_labels = flatten(test_features)

_train_features = np.array(_train_features)
print(_train_features.shape)
_train_features = np.reshape(_train_features, (_train_features.shape[0],
                                               _train_features.shape[1]*_train_features.shape[2]))
print(_train_features.shape)

_test_features = np.array(_test_features)
print(_test_features.shape)
_test_features = np.reshape(_test_features, (_test_features.shape[0],
                                             _test_features.shape[1]*_test_features.shape[2]))
print(_test_features.shape)

# encoded feature representation
# start
input_img = Input(shape=(_train_features.shape[1],))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(_train_features.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(_train_features, _train_features, epochs=50, batch_size=64, shuffle=True, verbose=0)

_train_features = encoder.predict(_train_features)
_test_features = encoder.predict(_test_features)

print(_train_features.shape)
print(_test_features.shape)

# end

model = KNeighborsClassifier(n_neighbors=k, weights='distance')
model.fit(_train_features, _train_labels)
_predict_labels = model.predict(_test_features)
f_score = metrics.f1_score(_test_labels, _predict_labels, average='macro')
accuracy = metrics.accuracy_score(_test_labels, _predict_labels)
results = 'pm,' + str(k) + ',' + str(accuracy)+',' + str(f_score)
print(results)
write_data(results_file, results)

_test_labels = pd.Series(_test_labels, name='Actual')
_predict_labels = pd.Series(_predict_labels, name='Predicted')
df_confusion = pd.crosstab(_test_labels, _predict_labels)
print(df_confusion)
write_data(results_file, str(df_confusion))
