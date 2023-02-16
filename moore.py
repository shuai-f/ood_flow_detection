import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from util import ml
from util.ml import simple_CNN

# Moore数据集使用网络监测器采集1 000个用户通过一条链路在24 h内的网络流量，
# 原始流量集包含在两个链路方向上连接节点的所有全双工流量。由于原始流量集太大，通过随机抽样方法将其划分为10个子集。
# 每个子集的采样时间几乎相同(每个大约1 680 s)，这些非重叠随机样本在24 h间隔内均匀分布。
# Moore数据集中每条网络流样本都是从一条完整的TCP双向流抽象出来，共包含249项特征，其中，最后一项特征是每条网络流相对应的类别。
# Moore流量数据集共有12种应用类型，由于GAMES类流量的数量太少，无法有效分类

labels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES',
          'INTERACTIVE', 'GAMES']
# train_labels = []
# ood_labels = []
# global train_labels, ood_labels, num_classes, num_pixels
#
# # Moore default configuration
# num_classes = 12
# num_pixels = 256

origin_dataset_root = '/inputs/moore_dataset/'
mid_dataset_root = '/mid/moore_dataset'


def init(num=1):
    """
    初始化Model参数
    :param num: 分割类型
    :return:
    """
    # global train_labels, ood_labels, num_classes, num_pixels
    numPixels = 256
    if num == 1:
        numClasses = 8
        trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', ]
        oodLabels = ['MULTIMEDIA', 'SERVICES', 'INTERACTIVE', 'GAMES', ]
    if num == 2:
        numClasses = 8
        trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'SERVICES', 'DATABASE', 'FTP-DATA', ]
        oodLabels = ['MULTIMEDIA', 'P2P', 'INTERACTIVE', 'GAMES', ]

    return numClasses, numPixels, trainLabels, oodLabels


# 数据预处理
# read the file,change 'Y,N,?,', translate to tensor
def data_preprocess(filename):
    """
    数据预处理
    :param filename: 数据集文件名
    :return:
    """
    X, Y, ood_X, ood_Y = [], [], [], []
    for f in filename:
        print(f)
        with open(os.getcwd() + origin_dataset_root + f, 'r') as file:

            for n, i in enumerate(file.readlines()[253:]):
                # print(n,i)
                i = i.replace('Y', '1')
                i = i.replace('N', '0')
                spl = i.split(',')
                if spl.count('?') > 8:
                    continue
                i = i.replace('\n', '')
                fz = [float(f) for f in i.split(',')[:-1] if f != '?']
                # 每一个样本占248维
                meana = sum(fz) / len(fz)
                i = i.replace('?', str(0))
                # 均值填充，加高斯白噪声
                # x = [float(j) for j in i.split(',')[:-1]] + [meana] * 8 + np.random.normal(0, 1, 256)
                x = [float(j) for j in i.split(',')[:-1]] + [0] * 8
                # x =x.tolist()
                y = i.split(',')[-1].replace('FTP-CO0TROL', 'FTP-CONTROL')
                y = y.replace('I0TERACTIVE', 'INTERACTIVE')

                #
                y = labels.index(y)
                X.append(x)
                Y.append(y)
            file.close()
    return X, Y


def splits(model_set_label, ood_set_label, filename):
    """
    分割数据集为 ood数据和非ood数据
    :param model_set_label: 模型训练类型
    :param ood_set_label: 分布外数据类型
    :param filename: 数据集文件名
    :return:
    """
    X, Y, ood_X, ood_Y = [], [], [], []
    for f in filename:
        print(f)
        with open(os.getcwd() + origin_dataset_root + f, 'r') as file:

            for n, i in enumerate(file.readlines()[253:]):
                # print(n,i)
                i = i.replace('Y', '1')
                i = i.replace('N', '0')
                spl = i.split(',')
                if spl.count('?') > 8:
                    continue
                i = i.replace('\n', '')
                fz = [float(f) for f in i.split(',')[:-1] if f != '?']
                # 每一个样本占248维
                meana = sum(fz) / len(fz)
                i = i.replace('?', str(0))
                # 均值填充，加高斯白噪声
                # x = [float(j) for j in i.split(',')[:-1]] + [meana] * 8 + np.random.normal(0, 1, 256)
                x = [float(j) for j in i.split(',')[:-1]] + [0] * 8
                # x =x.tolist()
                y = i.split(',')[-1].replace('FTP-CO0TROL', 'FTP-CONTROL')
                y = y.replace('I0TERACTIVE', 'INTERACTIVE')

                # split
                if y in ood_set_label:
                    y = ood_set_label.index(y)
                    ood_X.append(x)
                    ood_Y.append(y)
                else:
                    y = model_set_label.index(y)
                    X.append(x)
                    Y.append(y)
            file.close()
    return X, Y, ood_X, ood_Y


# data nomalization
num_classes, num_pixels, train_labels, ood_labels = init(1)
# train_x,train_y = data_prepross(['entry01.weka.allclass.arff',])
total_x, total_y, ood_x, ood_y = splits(train_labels, ood_labels, [
    'entry01.weka.allclass.arff', 'entry02.weka.allclass.arff', 'entry03.weka.allclass.arff',
    'entry04.weka.allclass.arff', 'entry05.weka.allclass.arff', 'entry06.weka.allclass.arff',
    'entry07.weka.allclass.arff', 'entry08.weka.allclass.arff', 'entry09.weka.allclass.arff',
    'entry10.weka.allclass.arff', ])

train_x, test_x, train_y, test_y = train_test_split(total_x, total_y, test_size=0.25, random_state=0)
train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
ood_x = tf.convert_to_tensor(ood_x, dtype=tf.float32)
ood_y = tf.convert_to_tensor(ood_y, dtype=tf.int32)
train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)
ood_x = tf.keras.utils.normalize(ood_x, axis=1)

print(train_x)
print(train_y)
print(ood_x)
print(ood_y)

if __name__ == '__main__':
    ml.config(8, 256, ood_x, ood_y, train_labels, ood_labels)
    print("\nCNN------------------------------------------------\n")
    simple_CNN(train_x, train_y, test_x, test_y)
    # print("\nBaseline------------------------------------------------\n")
    # baseline(train_x, train_y, test_x, test_y)
    # print("\nBayes------------------------------------------------\n")
    # Bayes(train_x, train_y, test_x, test_y)
    # print("\nDecisionTr------------------------------------------------\n")
    # DecisionTr(train_x, train_y, test_x, test_y)
    # print("\nKNN------------------------------------------------\n")
    # Knn(train_x.numpy(),train_y.numpy(),test_x.numpy(),test_y.numpy())
    # print("\nSVM------------------------------------------------\n")
    # SVM(train_x, train_y, test_x, test_y)
    # print("\n灰度------------------------------------------------\n")
    # plt_image(train_x, train_y)
    # print("\n混淆矩阵------------------------------------------------\n")
    # print("\nLSTM------------------------------------------------\n")
    # lstm(train_x, train_y, test_x, test_y)
    print("\nRandomForest------------------------------------------------\n")
    # RandomForest(train_x, train_y, test_x, test_y)
