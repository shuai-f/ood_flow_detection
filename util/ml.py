import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ood_detection import Msp, VirtualLogit
from util.utils import plot_confusion_matrix, plt_index_of_model, save_model, load_model

# Moore default configuration
num_classes = 12
num_pixels = 256
ood_x = []
ood_y = []
train_labels = []
ood_labels = []


def config(numClass, numPixels, ood_X, ood_Y, train_Labels, ood_Labels):
    global num_classes, num_pixels, ood_x, ood_y, train_labels, ood_labels
    num_classes = numClass
    num_pixels = numPixels
    ood_x = ood_X
    ood_y = ood_Y
    train_labels = train_Labels
    ood_labels = ood_Labels


# ,class_weight=class_weight
# BP算法
def baseline(train_x, train_y, test_x, test_y):
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


def simple_CNNmodel():
    model = keras.models.Sequential([
        layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', input_shape=(16, 16, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        # layers.Dropout(0.25),
        # (5,5,16) > 400
        # 生成一个一维向量
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        # 全连接层：特征提取器，将学到的特征表示映射到样本的标记空间，全连接一般会把卷积输出的二维特征图转化成一维的一个向量
        layers.Dense(num_classes, activation='softmax'),
        # layers.Dense(1, activation='softmax')
    ])
    # Compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model


# ,class_weight=class_weight
def simple_CNN(train_x, train_y, test_x, test_y):
    epochs = 17 # 25
    batch_size = 128
    t1 = time.time()
    model = simple_CNNmodel()
    # 数据
    X_train = tf.reshape(train_x, [-1, 16, 16, 1])
    X_test = tf.reshape(test_x, [-1, 16, 16, 1])
    # X_ood = tf.reshape(ood_x, [-1, 16, 16, 1])

    # 初始化模型获取第一层的权重
    # 打印模型摘要
    model.summary()
    # 模型训练
    # loss：训练集损失值，accuracy:训练集准确率，val_loss:测试集损失值，val_accruacy:测试集准确率
    history = model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=epochs, batch_size=batch_size,
                        verbose=2)
    # 模型评估 (loss, accuracy)
    scores = model.evaluate(X_test, test_y, verbose=0)
    t2 = time.time()

    # 模型保存
    # model.save_weights('output/weight/cnn_weights.h5')
    save_model(model, lis=ood_labels)

    # 训练模型后获取第一层的权重
    print("\nweights and bias : -------------------")
    weight_Dense_1, bias_Dense_1 = model.get_layer(index=-1).get_weights()
    print(weight_Dense_1)
    print(bias_Dense_1)
    print(weight_Dense_1.shape, bias_Dense_1.shape)
    # np.savetxt("./output/weight/cnn_weight.txt", weight_Dense_1, fmt="%d")
    # np.savetxt("./output/weight/cnn_bias.txt", bias_Dense_1, fmt="%d")
    # utils.save_weights(weight_Dense_1, bias_Dense_1, "CNN_weights")

    # predict函数：训练后返回一个概率值数组，此数组的大小为n·k，第i行第j列上对应的数值代表模型对此样本属于某类标签的概率值，行和为1
    predict_y = model.predict(X_test)
    # argmax是一种函数，是对函数求参数(集合)的函数。当我们有另一个函数y=f(x)时，若有结果x0= argmax(f(x))，则表示当函数f(x)取x=x0的时候，得到f(x)取值范围的最大值；若有多个点使得f(x)取得相同的最大值，那么argmax(f(x))的结果就是一个点集。
    pred_y = np.argmax(predict_y, axis=1)

    Msp(ood_x, ood_y, model, train_labels, ood_labels)

    # print("\nclassification : -----------------")
    # print(pred_y, pred_y.shape)
    print("Scores : ", scores)
    print("Baseline Error: %.2f%%, Total Time : %.2f" % ((100 - scores[1] * 100), t2 - t1))
    print(history.history)
    plt_index_of_model(epochs, history.history['loss'], history.history['accuracy'], history.history['val_loss'],
                       history.history['val_accuracy'])
    plot_confusion_matrix("CNN", test_y, pred_y, train_labels)
    return scores, pred_y


# CNN 模型验证与分布外检测
def load_CNN(train_x, train_y, test_x, test_y, ood_X=None, ood_Y=None):
    # 模型验证
    # 数据
    if ood_Y is None:
        ood_Y = ood_y
    if ood_X is None:
        ood_X = ood_x
    print("ood simple scale : %d" % len(ood_X))
    X_test = tf.reshape(test_x, [-1, 16, 16, 1])
    # X_ood = tf.reshape(ood_X, [-1, 16, 16, 1])
    model = simple_CNNmodel()
    # model.load_weights('output/weight/cnn_weights.h5')
    load_model(model, lis=ood_labels)
    # # 模型评估 (loss, accuracy)
    # scores = model.evaluate(X_test, test_y, verbose=0)
    # # predict函数：训练后返回一个概率值数组，此数组的大小为n·k，第i行第j列上对应的数值代表模型对此样本属于某类标签的概率值，行和为1
    # predict_y = model.predict(X_test)
    # # argmax是一种函数，是对函数求参数(集合)的函数。当我们有另一个函数y=f(x)时，若有结果x0= argmax(f(x))，则表示当函数f(x)取x=x0的时候，得到f(x)取值范围的最大值；若有多个点使得f(x)取得相同的最大值，那么argmax(f(x))的结果就是一个点集。
    # pred_y = np.argmax(predict_y, axis=1)
    # print("Scores : ", scores)
    # print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
    # plot_confusion_matrix("CNN", test_y, pred_y, train_labels)

    # Baseline评估
    # Msp(test_x, test_y, ood_X, ood_Y, model, train_labels, ood_labels)

    # Virtual Logit评估
    # print("Virtual Logit : ==================================")
    VirtualLogit(ood_X, ood_Y, train_x, train_y,test_x, test_y, model, train_labels, ood_labels)

    return


def Bayes(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    mnb = GaussianNB()  #
    mnb.fit(trainData, trainLabel)  #
    y_predict = mnb.predict(testData)
    t2 = time.time()
    print(t2 - t1)
    print('The Accuracy of Naive Bayes Classifier is:', mnb.score(testData, testLabel))
    print(y_predict, y_predict.shape)
    plot_confusion_matrix("Bayes", testLabel, y_predict)


def DecisionTr(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    model = DecisionTreeClassifier()
    model.fit(trainData, trainLabel)
    predicted = model.predict(testData)
    score = metrics.accuracy_score(testLabel, predicted)
    t2 = time.time()
    print(t2 - t1, score)
    plot_confusion_matrix("Decision Tree", testLabel, predicted)


def SVM(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    clf = SVC()
    clf.fit(trainData, trainLabel)
    svmPredict = clf.predict(testData)
    svmScore = metrics.accuracy_score(testLabel, svmPredict)
    t2 = time.time()
    print(t2 - t1, svmScore)
    plot_confusion_matrix("SVM", testLabel, svmPredict)


def Knn(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    knn = KNeighborsClassifier()
    knn.fit(trainData, trainLabel)
    knnPredict = knn.predict(testData)
    knnscore = metrics.accuracy_score(testLabel, knnPredict)
    t2 = time.time()
    print(t2 - t1, knnscore)
    plot_confusion_matrix("KNN", testLabel, knnPredict)


def RandomForest(trainData, trainLabel, testData, testLabel):
    t1 = time.time()
    rf = RandomForestClassifier()
    rf.fit(trainData, trainLabel)
    rf_predict = rf.predict(testData)
    rf_score = metrics.accuracy_score(testLabel, rf_predict)
    t2 = time.time()
    print(t2 - t1, rf_score)
    plot_confusion_matrix("RandomForest", testLabel, rf_predict)


# 要求输入dimension = 3D， shape = (simple, timesteps, feature)
def lstm(train_x, train_y, test_x, test_y):
    def lstmModel():
        model = keras.models.Sequential([
            # layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', input_shape=(16, 16, 1), activation='relu'),
            # layers.Conv2D(64, (3, 3), input_shape=(3, 32, 32), padding='same', ),
            # layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            # layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
            # layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            # (5,5,16) > 400
            # layers.Flatten(),
            layers.Reshape([16, 16]),
            layers.LSTM(units=256, return_sequences=True),
            layers.Flatten(),
            # layers.Dense(256, activation='relu'),
            # layers.Dropout(0.5),
            # # layers.Dense(84, activation='relu'),
            # layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),
        ])
        # Compile model
        model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
        # model.build()
        return model

    t1 = time.time()
    model = lstmModel()
    X_train = tf.reshape(train_x, [-1, 16, 16, 1])
    print(X_train.shape)
    X_test = tf.reshape(test_x, [-1, 16, 16, 1])
    # X_train = tf.reshape(train_x, [-1, train_x.shape[1], 1])
    # X_test = tf.reshape(test_x, [-1, train_x.shape[1],  1])
    # model.build(X_train.shape)

    # model.summary()
    # batch_size = 32, 64, 128
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=25, batch_size=128, verbose=2)
    scores = model.evaluate(test_x, test_y, verbose=0)
    t2 = time.time()
    pred_y = model.predict(test_x)
    print(scores)
    print("LSTM Error: %.2f%%" % (100 - scores[1] * 100), t2 - t1)
    print(history.history)
    pred_y = np.argmax(pred_y, axis=1)
    plot_confusion_matrix("LSTM", test_y, pred_y)
    return scores
