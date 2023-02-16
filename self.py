#
# # 自采集数据流量
#
# import os
# import time
#
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
#
# from util import ml
# from util.ml import baseline, simple_CNN, Bayes, DecisionTr, Knn, SVM, RandomForest, lstm
# from util.utils import plt_image
#
# # 自建数据集
#
# labels = ['txsp',]
#
# # Moore default configuration
# num_classes = 12
# num_pixels = 256
#
# # 数据预处理
# # read the file,change 'Y,N,?,', translate to tensor
# def data_preprocess(filename):
#     X, Y = [], []
#     for f in filename:
#         print(f)
#         with open(os.getcwd() + '/inputs/self_dataset/' + f, 'r') as file:
#
#             for n, i in enumerate(file.readlines()[253:]):
#                 # print(n,i)
#                 i = i.replace('Y', '1')
#                 i = i.replace('N', '0')
#                 spl = i.split(',')
#                 if spl.count('?') > 8:
#                     continue
#                 i = i.replace('\n', '')
#                 fz = [float(f) for f in i.split(',')[:-1] if f != '?']
#                 # 每一个样本占248维
#                 meana = sum(fz) / len(fz)
#                 i = i.replace('?', str(0))
#                 # 均值填充，加高斯白噪声
#                 # x = [float(j) for j in i.split(',')[:-1]] + [meana] * 8 + np.random.normal(0, 1, 256)
#                 x = [float(j) for j in i.split(',')[:-1]] + [0] * 8
#                 # x =x.tolist()
#                 y = i.split(',')[-1].replace('FTP-CO0TROL', 'FTP-CONTROL')
#                 y = y.replace('I0TERACTIVE', 'INTERACTIVE')
#                 y = labels.index(y)
#                 X.append(x)
#                 Y.append(y)
#             file.close()
#     return X, Y
#
#
# # data nomalization
# # train_x,train_y = data_prepross(['entry01.weka.allclass.arff',])
# start = time.time()
# total_x, total_y = data_preprocess([
#      'entry01.weka.allclass.arff', 'entry02.weka.allclass.arff', 'entry03.weka.allclass.arff',
#      'entry04.weka.allclass.arff','entry05.weka.allclass.arff', 'entry06.weka.allclass.arff',
#      'entry07.weka.allclass.arff', 'entry08.weka.allclass.arff','entry09.weka.allclass.arff',
#      'entry10.weka.allclass.arff',])
# end = time.time()
# print()
#
# train_x, test_x, train_y, test_y = train_test_split(total_x, total_y, test_size=0.25, random_state=0)
# train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
# train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
# test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
# test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
# train_x = tf.keras.utils.normalize(train_x, axis=1)
# test_x = tf.keras.utils.normalize(test_x, axis=1)
# print(train_x)
# print(train_y)
#
# if __name__ == '__main__':
#     ml.config(numClass=num_classes, numPixels=num_pixels)
#     print("\nBaseline------------------------------------------------\n")
#     baseline(train_x, train_y, test_x, test_y)
#     print("\nCNN------------------------------------------------\n")
#     simple_CNN(train_x, train_y, test_x, test_y)
#     print("\nBayes------------------------------------------------\n")
#     Bayes(train_x, train_y, test_x, test_y)
#     print("\nDecisionTr------------------------------------------------\n")
#     DecisionTr(train_x, train_y, test_x, test_y)
#     print("\nKNN------------------------------------------------\n")
#     Knn(train_x.numpy(),train_y.numpy(),test_x.numpy(),test_y.numpy())
#     print("\nSVM------------------------------------------------\n")
#     SVM(train_x, train_y, test_x, test_y)
#     print("\nBayes------------------------------------------------\n")
#     RandomForest(train_x, train_y, test_x, test_y)
#     print("\n灰度------------------------------------------------\n")
#     plt_image(train_x, train_y)
#     print("\n混淆矩阵------------------------------------------------\n")
#     print("\nLSTM------------------------------------------------\n")
#     lstm(train_x, train_y, test_x, test_y)