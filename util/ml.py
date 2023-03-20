import joblib
import tensorflow as tf
import numpy as np
import time

from keras.callbacks import Callback
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from util.ood_detection import VirtualLogit, LocalThreshold
from util.utils import plot_confusion_matrix, plt_index_of_model, save_model, load_model, plt_feat_importance, \
    list_to_str, read_features

# Moore default configuration
num_classes = 12
num_pixels = 256
ood_x = []
ood_y = []
train_labels = []
ood_labels = []
input_npy_moore_dir = './inputs/npy/moore/'
weight_dir = './output/weight/'
features = read_features()

class Metrics(Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return

metrics_ = Metrics()

def read_data(oodLabels=None):
    if oodLabels is None:
        oodLabels = ood_labels
    splits_dir = input_npy_moore_dir + list_to_str(oodLabels) + '_'
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
    splits_dir = input_npy_moore_dir + list_to_str(oodLabels) + '_'
    np.save(splits_dir + 'train_x.npy', train_x)
    np.save(splits_dir + 'train_y.npy', train_y)
    np.save(splits_dir + 'test_x.npy', test_x)
    np.save(splits_dir + 'test_y.npy', test_y)
    np.save(splits_dir + 'ood_x.npy', ood_x)
    np.save(splits_dir + 'ood_y.npy', ood_y)

def config(numClass, numPixels, ood_X, ood_Y, train_Labels, ood_Labels):
    global num_classes, num_pixels, ood_x, ood_y, train_labels, ood_labels
    num_classes = numClass
    num_pixels = numPixels
    ood_x = ood_X
    ood_y = ood_Y
    train_labels = train_Labels
    ood_labels = ood_Labels

def simple_CNNmodel():
    model = keras.models.Sequential([
        # layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', input_shape=(16, 16, 1), activation='relu'),
        # layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        # layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
        # layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        # layers.Dropout(0.25),
        # (5,5,16) > 400
        # 生成一个一维向量
        # layers.Flatten(),
        # layers.Dense(256, activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(128, activation='relu'),
        # 全连接层：特征提取器，将学到的特征表示映射到样本的标记空间，全连接一般会把卷积输出的二维特征图转化成一维的一个向量
        layers.Dense(num_classes, activation='softmax'),
        # layers.Dense(1, activation='softmax')
    ])
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=[None, 16 * 16, ])
    return model

# ,class_weight=class_weight
def simple_CNN(train_x, train_y, test_x, test_y):
    """
    CNN 模型训练
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    epochs = 30  # 25
    batch_size = 128
    t1 = time.time()
    model = simple_CNNmodel()
    # 数据写入
    write_data(train_x, train_y, test_x, test_y, ood_x, ood_y)
    # 数据处理
    X_train = tf.reshape(train_x, [-1, 16*16, 1])
    X_test = tf.reshape(test_x, [-1, 16*16, 1])
    # X_ood = tf.reshape(ood_x, [-1, 16, 16, 1])

    # model.build(input_shape=[None,16*16,])
    # 打印模型摘要
    model.summary()
    # 模型训练
    # loss：训练集损失值，accuracy:训练集准确率，val_loss:测试集损失值，val_accruacy:测试集准确率
    history = model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=epochs, batch_size=batch_size,
                        verbose=2,)
    # 模型评估 (loss, accuracy)
    scores = model.evaluate(X_test, test_y, verbose=0)
    t2 = time.time()

    # 模型保存
    # model.save_weights('output/weight/cnn_weights.h5')
    save_model(model, lis=ood_labels)

    # predict函数：训练后返回一个概率值数组，此数组的大小为n·k，第i行第j列上对应的数值代表模型对此样本属于某类标签的概率值，行和为1
    predict_y = model.predict(X_test)
    # argmax是一种函数，是对函数求参数(集合)的函数。当我们有另一个函数y=f(x)时，若有结果x0= argmax(f(x))，则表示当函数f(x)取x=x0的时候，得到f(x)取值范围的最大值；若有多个点使得f(x)取得相同的最大值，那么argmax(f(x))的结果就是一个点集。
    pred_y = np.argmax(predict_y, axis=1)

    # Msp(ood_x, ood_y, model, train_labels, ood_labels)

    print("Scores : ", scores)
    print("Baseline Error: %.2f%%, Total Time : %.2f" % ((100 - scores[1] * 100), t2 - t1))
    print(history.history)
    plt_index_of_model(epochs, history.history['loss'], history.history['accuracy'], history.history['val_loss'],
                       history.history['val_accuracy'])
    plot_confusion_matrix("CNN", test_y, pred_y, train_labels)
    return scores, pred_y

def load_CNN(train_x, train_y, test_x, test_y, ood_X=None, ood_Y=None):
    """
    CNN 模型验证与分布外检测
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param ood_X:
    :param ood_Y:
    :return:
    """
    # 模型验证
    # 数据
    if ood_Y is None:
        ood_Y = ood_y
    if ood_X is None:
        ood_X = ood_x
    print("ood simple scale : %d" % len(ood_X))
    X_test = tf.reshape(test_x, [-1, 16, 16, 1])
    model = simple_CNNmodel()
    load_model(model, lis=ood_labels)
    # Baseline评估
    # Msp(test_x, test_y, ood_X, ood_Y, model, train_labels, ood_labels)

    # 获取模型权重和偏置
    print("\nweights and bias : -------------------")
    # weight_Dense_1, bias_Dense_1 = model.get_layer(index=-3).get_weights()
    # weight_Dense_2, bias_Dense_2 = model.get_layer(index=-2).get_weights()
    # weight_Dense_1 = weight_Dense_1 @ weight_Dense_2
    # bias_Dense_1 = bias_Dense_1 @ weight_Dense_2 + bias_Dense_2
    # # weight_Dense_1, bias_Dense_1 = model.get_layer(index=-2).get_weights()
    # # print(weight_Dense_1.shape, bias_Dense_1.shape)
    # weight_Dense_2, bias_Dense_2 = model.get_layer(index=-1).get_weights()
    # # print(weight_Dense_2.shape, bias_Dense_2.shape)
    # # W ∈ RN×C
    # w = weight_Dense_1 @ weight_Dense_2
    # w = w.T
    # # b ∈ RC
    # b = bias_Dense_1 @ weight_Dense_2 + bias_Dense_2
    w, b = model.get_layer(index=-1).get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)

    # Virtual Logit评估
    VirtualLogit(ood_X, ood_Y, train_x, train_y, test_x, test_y, model, w, b, train_labels, ood_labels)

    return

def contrast_learning_CNN():
    """
    对比学习应用于所有训练数据集，
    :return:
    """

    # 数据
    train_x, train_y, test_x, test_y, ood_x, ood_y = read_data(ood_labels)
    X_train = tf.reshape(train_x, [-1, 16, 16, 1])
    X_test = tf.reshape(test_x, [-1, 16, 16, 1])
    # X_ood = tf.reshape(ood_X, [-1, 16, 16, 1])
    model = simple_CNNmodel()
    # model.load_weights('output/weight/cnn_weights.h5')
    # 初始IND分类模型M
    load_model(model, lis=ood_labels)
    model.summary()

    # 训练样本结果
    predict_y = model.predict(X_train)
    pred_y = np.argmax(predict_y, axis=1)
    max_softmax = np.max(predict_y, axis=1)
    accuracy = accuracy_score(train_y, pred_y, )
    print("accuracy : %.2f%%" % (accuracy * 100))
    # 混淆对
    res = [i for i in range(len(pred_y)) if pred_y[i] != train_y[i]]
    print("src -> tgt: ")
    print([[(train_labels[train_y[i]], train_labels[pred_y[i]]), max_softmax[i]] for i in res])
    confused_map = {}
    for i in res:
        if confused_map.get((train_labels[train_y[i]], train_labels[pred_y[i]])) is None:
            confused_map[(train_labels[train_y[i]], train_labels[pred_y[i]])] = [max_softmax[i]]
        else :
            confused_map[(train_labels[train_y[i]], train_labels[pred_y[i]])].append(max_softmax[i])
    with open('./output/splits/splits_2/confused_label.csv', 'w') as f:
        f.write("src_label,tgt_label,num,val_prob\n")
        for key, value in confused_map.items():
            f.write("{},{},{},{}\n".format(key[0], key[1], len(value), np.mean(value)))
    # 容易混淆类别，区分
    confused_labels = set()
    for i in res:
        confused_labels.add(train_y[i])
        confused_labels.add(pred_y[i])
    print("confused label set :")
    print(confused_labels)  # all 混淆
    # for i in range(train_y.shape[0]):  # 行索引
    #     if train_y[i] not in confused_labels:
    #         part_1 = X_train[0:i - 1, :]
    #         part_2 = X_train[i + 1:, :]
    #         X_train = tf.concat([part_1, part_2], 0)

    # 获取中间层权重和偏置
    weight_Dense_1, bias_Dense_1 = model.get_layer(index=-1).get_weights()
    weight_Dense_1 = weight_Dense_1.T  # 128 x 256

    # LDA降维，
    DIM = 2
    data_2 = LinearDiscriminantAnalysis(n_components=DIM).fit_transform(train_x, train_y)

    # 二维
    plt.figure(figsize=(4,4))
    plt.plot(122)
    plt.title("LDA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = train_y)
    plt.savefig("LDA.png",dpi=600)
    plt.show()
    #
    # # 三维
    # # from mpl_toolkits.mplot3d import Axes3D
    # # fig = plt.figure(figsize=(10, 10))
    # # ax = fig.add_subplot(111, projection='3d')
    # #
    # # ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], c=train_y)
    # # plt.legend(loc=0)
    # # plt.savefig("LDA_dim3.png", dpi=600)
    # # plt.show()
    #
    #
    # lda=LinearDiscriminantAnalysis()
    # lda.fit(train_x, train_y)
    # w = lda.coef_
    # b = lda.intercept_
    # print('Coefficients:%s, intercept %s' % (lda.coef_, lda.intercept_))  # 输出权重向量和b
    # print(w.shape, b.shape)
    # print('Score: %.2f' % lda.score(test_x, test_y))  # 测试集
    # print('Score: %.2f' % lda.score(train_x, train_y))  # 训练集
    #
    # # DecisionTr(data_2, train_y, data_test, test_y)
    # train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
    # train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
    # test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    # test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
    # ood_x = tf.convert_to_tensor(ood_x, dtype=tf.float32)
    # ood_y = tf.convert_to_tensor(ood_y, dtype=tf.int32)
    # VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, lda, w, b, train_labels, ood_labels)

    # # 模型训练
    # epochs = 5
    # batch_size = 128
    # # tf.keras.losses.
    # # transfer one hot
    # Y_train = tf.one_hot(train_y, depth=num_classes)
    # Y_test = tf.one_hot(test_y, depth=num_classes)
    # # model = simple_CNNmodel()
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size,
    #                     verbose=2, callbacks=[])
    #
    # # predict函数：训练后返回一个概率值数组，此数组的大小为n·k，第i行第j列上对应的数值代表模型对此样本属于某类标签的概率值，行和为1
    # predict_y = model.predict(X_test)
    # # argmax是一种函数，是对函数求参数(集合)的函数。当我们有另一个函数y=f(x)时，若有结果x0= argmax(f(x))，则表示当函数f(x)取x=x0的时候，得到f(x)取值范围的最大值；若有多个点使得f(x)取得相同的最大值，那么argmax(f(x))的结果就是一个点集。
    # pred_y = np.argmax(predict_y, axis=1)
    # accuracy = accuracy_score(test_y, pred_y, )
    # print("accuracy : %.2f%%" % (accuracy * 100))

def local_threshold(train_x, train_y, test_x, test_y, ood_X=None, ood_Y=None):
    if ood_Y is None:
        ood_Y = ood_y
    if ood_X is None:
        ood_X = ood_x
    print("ood simple scale : %d" % len(ood_X))
    X_test = tf.reshape(test_x, [-1, 16, 16, 1])
    model = simple_CNNmodel()
    load_model(model, lis=ood_labels)
    # Baseline评估
    # Msp(test_x, test_y, ood_X, ood_Y, model, train_labels, ood_labels)

    # 获取模型权重和偏置
    print("\nweights and bias : -------------------")
    # weight_Dense_1, bias_Dense_1 = model.get_layer(index=-3).get_weights()
    # weight_Dense_2, bias_Dense_2 = model.get_layer(index=-2).get_weights()
    # weight_Dense_1 = weight_Dense_1 @ weight_Dense_2
    # bias_Dense_1 = bias_Dense_1 @ weight_Dense_2 + bias_Dense_2
    # # weight_Dense_1, bias_Dense_1 = model.get_layer(index=-2).get_weights()
    # # print(weight_Dense_1.shape, bias_Dense_1.shape)
    # weight_Dense_2, bias_Dense_2 = model.get_layer(index=-1).get_weights()
    # # print(weight_Dense_2.shape, bias_Dense_2.shape)
    # # W ∈ RN×C
    # w = weight_Dense_1 @ weight_Dense_2
    # w = w.T
    # # b ∈ RC
    # b = bias_Dense_1 @ weight_Dense_2 + bias_Dense_2
    w,b = model.get_layer(index=-1).get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)

    # Virtual Logit评估
    LocalThreshold(ood_X, ood_Y, train_x, train_y, test_x, test_y, model, w, b, train_labels, ood_labels, 'Energy')

####################################################
##                                                ##
##              Other ML Methods                  ##
##                                                ##
####################################################

# ,class_weight=class_weight
def baseline(train_x, train_y, test_x, test_y):
    """
    BP
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """

    def baseline_model():
        model = Sequential()

        model.add(layers.Dense(num_pixels, input_dim=num_pixels, activation='relu'))
        # layers.Dropout(0.5)
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    t1 = time.time()
    model = baseline_model()
    X_train = tf.reshape(train_x, [-1, 256])
    X_test = tf.reshape(test_x, [-1, 256])
    history = model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=20, batch_size=200, verbose=2, )
    scores = model.evaluate(X_test, test_y, verbose=0)
    pred_y = model.predict(X_test)
    print(pred_y, pred_y.shape)
    # argmax是一种函数，是对函数求参数(集合)的函数。当我们有另一个函数y=f(x)时，若有结果x0= argmax(f(x))，则表示当函数f(x)取x=x0的时候，得到f(x)取值范围的最大值；若有多个点使得f(x)取得相同的最大值，那么argmax(f(x))的结果就是一个点集。
    pred_y = np.argmax(pred_y, axis=1)
    print(pred_y, pred_y.shape)
    t2 = time.time()
    print("Scores : ", scores)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100), t2 - t1)
    print(history.history)
    plot_confusion_matrix("BP -- Baseline", test_y, pred_y)
    return scores, pred_y

def Bayes(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    mnb = GaussianNB()  #
    mnb.fit(trainData, trainLabel)  #
    y_predict = mnb.predict(testData)
    t2 = time.time()
    precision, recall, F1, _ = precision_recall_fscore_support(testLabel, y_predict, average="micro")
    print("精准率: {0:.2f}, 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))
    print(t2 - t1)
    print('The Accuracy of Naive Bayes Classifier is:', mnb.score(testData, testLabel))
    print(y_predict, y_predict.shape)
    plot_confusion_matrix("Bayes", testLabel, y_predict)

def DecisionTr(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    # 数据
    model = DecisionTreeClassifier(class_weight='balanced')
    model.fit(trainData, trainLabel)
    predicted = model.predict(testData)
    score = metrics.accuracy_score(testLabel, predicted)
    t2 = time.time()
    precision, recall, F1, _ = precision_recall_fscore_support(testLabel, predicted, average="micro")
    print("精准率: {0:.2f}, 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))
    plt_feat_importance(model, 15)
    # 模型保存
    joblib.dump(model, weight_dir + 'DecisionTree.pkl')

    print(t2 - t1, score)
    plot_confusion_matrix("Decision Tree", testLabel, predicted, train_labels)

def load_DecisionTr():
    # 数据
    train_x, train_y, test_x, test_y, ood_x, ood_y = read_data(ood_labels)
    model = joblib.load(weight_dir + 'DecisionTree.pkl')
    predicted = model.predict(test_x)
    score = metrics.accuracy_score(test_y, predicted)
    precision, recall, F1, _ = precision_recall_fscore_support(test_y, predicted, average="micro")
    print("精准率: {0:.2f}, 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))
    print("Score : {}".format(score))

    w = model.coef_
    b = model.intercept_
    VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, model, model, w, b, train_labels, ood_labels)

def SVM(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    clf = SVC()
    clf.fit(trainData, trainLabel)
    svmPredict = clf.predict(testData)
    svmScore = metrics_.accuracy_score(testLabel, svmPredict)
    t2 = time.time()
    print(t2 - t1, svmScore)
    plot_confusion_matrix("SVM", testLabel, svmPredict)

def Knn(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    knn = KNeighborsClassifier()
    knn.fit(trainData, trainLabel)
    knnPredict = knn.predict(testData)
    knnscore = metrics_.accuracy_score(testLabel, knnPredict)
    t2 = time.time()
    print(t2 - t1, knnscore)
    plot_confusion_matrix("KNN", testLabel, knnPredict)

def RandomForest(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    rf = RandomForestClassifier()
    rf.fit(trainData, trainLabel)
    rf_predict = rf.predict(testData)
    rf_score = metrics_.accuracy_score(testLabel, rf_predict)
    t2 = time.time()
    precision, recall, F1, _ = precision_recall_fscore_support(testLabel, rf_predict, average="binary")
    print("精准率: {0:.2f}. 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))
    print(t2 - t1, rf_score)
    plot_confusion_matrix("RandomForest", testLabel, rf_predict)

# 要求输入dimension = 3D， shape = (simple, timesteps, feature)
def LSTM(train_x, train_y, test_x, test_y):
    def lstmModel():
        model = keras.models.Sequential([
            layers.LSTM(128),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])
        # Compile model
        model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
        # model.build()
        return model

    t1 = time.time()
    model = lstmModel()
    X_train = tf.reshape(train_x, [-1, 256, 1])
    print(X_train.shape)
    X_test = tf.reshape(test_x, [-1, 256, 1])
    # X_train = tf.reshape(train_x, [-1, train_x.shape[1], 1])
    # X_test = tf.reshape(test_x, [-1, train_x.shape[1],  1])
    # model.build(X_train.shape)

    # model.summary()
    # batch_size = 32, 64, 128
    history = model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=25, batch_size=128, verbose=2)
    scores = model.evaluate(test_x, test_y, verbose=0)
    t2 = time.time()
    pred_y = model.predict(test_x)
    print(scores)
    print("LSTM Error: %.2f%%" % (100 - scores[1] * 100), t2 - t1)
    print(history.history)
    pred_y = np.argmax(pred_y, axis=1)
    plot_confusion_matrix("LSTM", test_y, pred_y)
    return scores
