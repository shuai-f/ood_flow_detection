import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from util.utils import list_to_str

self_labels = ['百度地图','QQ音乐','中通快递','今日头条','华商头条','微信','百合婚恋','知乎','腾讯视频','艺龙旅行','菜鸟裹裹']
self_labels_abbreviation = ['bddt','qqyy','ztkd','jrtt','hstt','wx','bhhl','zh','txsp','yllx','cngg']

dataset_root = './inputs/self_dataset/'
feature_root = './inputs/self_dataset/features/'
feature_all = feature_root + 'entry.csv'


def init(num=1):
    """
    初始化Model参数
    :param num: 分割类型
    :return:
    """
    # global train_labels, ood_labels, num_classes, num_pixels
    numPixels = 81
    if num == 1:
        numClasses = 8
        trainLabels = ['bddt','qqyy','ztkd','jrtt','hstt','wx','bhhl', 'zh', ]
        oodLabels = ['txsp','yllx','cngg' ]
    if num == 2:
        numClasses = 7
        trainLabels = ['bddt', 'qqyy', 'ztkd', 'jrtt', 'hstt', 'wx', 'bhhl', ]
        oodLabels = ['zh', 'txsp', 'yllx', 'cngg']

    return numClasses, numPixels, trainLabels, oodLabels


def write_to_file():
    path = feature_root
    if not os.path.exists(path):
        os.makedirs(path)
        print('文件夹创建完成  ' + path)
    for i in range(len(self_labels)):
        f = dataset_root + self_labels[i] + '/' + self_labels_abbreviation[i] + '.csv'
        shutil.copy(f, path)

input_npy_self_dir = './inputs/npy/self/'

def read_data(oodLabels=None):
    if oodLabels is None:
        oodLabels = ood_labels
    splits_dir = input_npy_self_dir + list_to_str(oodLabels) + '_'
    train_x = np.load(splits_dir + 'train_x.npy')
    train_y = np.load(splits_dir + 'train_y.npy')
    test_x = np.load(splits_dir + 'test_x.npy')
    test_y = np.load(splits_dir + 'test_y.npy')
    ood_x = np.load(splits_dir + 'ood_x.npy')
    ood_y = np.load(splits_dir + 'ood_y.npy')
    return train_x, train_y, test_x, test_y, ood_x, ood_y

def write_data(train_x, train_y, test_x, test_y, ood_x, ood_y, oodLabels=None):
    if oodLabels is None:
        oodLabels = ood_labels
    splits_dir = input_npy_self_dir + list_to_str(oodLabels) + '_'
    np.save(splits_dir + 'train_x.npy', train_x)
    np.save(splits_dir + 'train_y.npy', train_y)
    np.save(splits_dir + 'test_x.npy', test_x)
    np.save(splits_dir + 'test_y.npy', test_y)
    np.save(splits_dir + 'ood_x.npy', ood_x)
    np.save(splits_dir + 'ood_y.npy', ood_y)

def read_features(train_labels, ood_labels):
    X, Y, oodX, oodY= [], [], [], []
    # fw = open(feature_all, 'a')
    for index in range(len(train_labels)): # 训练集Label
        label = train_labels[index]
        dir_index = self_labels_abbreviation.index(label)
        f = dataset_root + self_labels[dir_index] + '/' + label + '.csv'
        with open(f, 'r') as file:
            for n, i in enumerate(file.readlines()[1:]):
                i = i.replace('\n', '')
                spl = i.split(',')
                spl[0:6] = '0' # 五元组+timestamp
                spl.append('0') # 9*9
                x = [float(j) for j in spl]
                y = index
                X.append(x)
                Y.append(y)
                # fw.write(i + ',' + self_labels_abbreviation[index])
    for index in range(len(ood_labels)): # 训练集Label
        label = ood_labels[index]
        dir_index = self_labels_abbreviation.index(label)
        f = dataset_root + self_labels[dir_index] + '/' + label + '.csv'
        with open(f, 'r') as file:
            for n, i in enumerate(file.readlines()[1:]):
                i = i.replace('\n', '')
                spl = i.split(',')
                spl[0:6] = '0' # 五元组+timestamp
                spl.append('0') # 9*9
                x = [float(j) for j in spl]
                y = index
                oodX.append(x)
                oodY.append(y)
                # fw.write(i + ',' + self_labels_abbreviation[index])
    return X, Y, oodX, oodY

num_classes, num_pixels, train_labels, ood_labels = init(1)
total_x, total_y, ood_x, ood_y = read_features(train_labels, ood_labels)

train_x, test_x, train_y, test_y = train_test_split(total_x, total_y, test_size=0.25, random_state=0)
# train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
# train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
# test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
# test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
# ood_x = tf.convert_to_tensor(ood_x, dtype=tf.float32)
# ood_y = tf.convert_to_tensor(ood_y, dtype=tf.int32)
train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)
ood_x = tf.keras.utils.normalize(ood_x, axis=1)

if __name__ == '__main__':
    # write_to_file()
    write_data(train_x, train_y, test_x, test_y, ood_x, ood_y, ood_labels)

