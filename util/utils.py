import os
import time

import numpy as np
from matplotlib import pyplot as plt, axes
from sklearn.metrics import confusion_matrix

# Moore 数据集
labels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', 'MULTIMEDIA', 'SERVICES',
          'INTERACTIVE', 'GAMES']


def write_file(filename, mode, content):
    with open(filename, mode) as fp:
        fp.write(content)


def tensor_to_str(tensor):
    tensor = tensor.numpy().tolist()
    strNums = [str(x_i) for x_i in tensor]
    str_ = ",".join(strNums)
    return str_


def origin_data(x_train, y_train, x_test, y_test, title='default'):
    filename = "./inputs/origin_data"
    mode = "a"
    with open(filename, mode) as fp:
        fp.write("\ntitle:" + title)
        fp.write("\nX_train:\t" + str(x_train.shape) + "\n")
        fp.write("\nY_train:\t" + str(y_train.shape) + "\n")
        fp.write("\nX_test:\t" + str(x_test.shape) + "\n")
        fp.write("\nY_test:\t" + str(y_test.shape) + "\n")

        fp.write("\nX_train:\t" + str(x_train.shape) + "\n")
        fp.write(tensor_to_str(x_train))
        fp.write("\nY_train:\t" + str(y_train.shape) + "\n")
        fp.write(tensor_to_str(y_train))
        fp.write("\nX_test:\t" + str(x_test.shape) + "\n")
        fp.write(tensor_to_str(x_test))
        fp.write("\nY_test:\t" + str(y_test.shape) + "\n")
        fp.write(tensor_to_str(y_test))


def read_features():
    with open('./inputs/moore_dataset/features.txt', 'r') as f:
        features = f.readlines()
    return features


def list_to_str(lis):
    res = ''
    for i in lis:
        res += str(i) + '_'
    return res


def save_model(model, name='', lis=None):
    if model is None:
        return
    if name == '':
        name = list_to_str(lis) + 'weights.h5'
    print(name)
    model.save_weights('./output/weight/' + name)


def load_model(model, name='', lis=None):
    if model is None:
        return
    if name == '':
        name = list_to_str(lis) + 'weights.h5'
    model.load_weights('./output/weight/' + name)
    return model

def get_path():
    time_now = time.strftime("%Y-%m-%d", time.localtime())
    path = "./output/preference/" + '/' + time_now
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/figure'
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/' + time.strftime("%H-%M-%S", time.localtime()) + '_'
    return path

def save_weights(w, b, filename):
    weight_path = "./output/weight/"
    with open(weight_path + filename, 'w') as fp:
        fp.write("\nw:\t" + str(w.shape) + "\n")
        fp.write(tensor_to_str(w))
        fp.write("\nb:\t" + str(b.shape) + "\n")
        fp.write(tensor_to_str(b))


# 灰度图片
def plt_image(trainx, trainy):
    p_www = np.where(trainy == 0)[0][0]
    p_mail = np.where(trainy == 1)[0][0]
    p_control = np.where(trainy == 2)[0][0]
    p_pasv = np.where(trainy == 3)[0][0]
    p_attack = np.where(trainy == 4)[0][0]
    p_p2p = np.where(trainy == 5)[0][0]
    p_database = np.where(trainy == 6)[0][0]
    p_data = np.where(trainy == 7)[0][0]
    p_multimedia = np.where(trainy == 8)[0][0]
    p_service = np.where(trainy == 9)[0][0]
    p_interactive = np.where(trainy == 10)[0][0]
    p_games = np.where(trainy == 11)[0][0]
    plt.figure(num='classffication', figsize=(6, 12))
    plt.subplot(3, 4, 1)
    plt.title('WWW')
    plt.imshow(np.reshape(trainx[p_www], (16, 16)))
    plt.subplot(3, 4, 2)
    plt.title('MAIL')
    plt.imshow(np.reshape(trainx[p_mail], (16, 16)))
    plt.subplot(3, 4, 3)
    plt.title('FTP-CONTROL')
    plt.imshow(np.reshape(trainx[p_control], (16, 16)))
    plt.subplot(3, 4, 4)
    plt.title('FTP-PASV')
    plt.imshow(np.reshape(trainx[p_pasv], (16, 16)))
    plt.subplot(3, 4, 5)
    plt.title('ATTCK')
    plt.imshow(np.reshape(trainx[p_attack], (16, 16)))
    plt.subplot(3, 4, 6)
    plt.title('P2P')
    plt.imshow(np.reshape(trainx[p_p2p], (16, 16)))
    plt.subplot(3, 4, 7)
    plt.title('DATABASE')
    plt.imshow(np.reshape(trainx[p_database], (16, 16)))
    plt.subplot(3, 4, 8)
    plt.title('FTP-DATA')
    plt.imshow(np.reshape(trainx[p_data], (16, 16)))
    plt.subplot(3, 4, 9)
    plt.title('MULTIMEDIA')
    plt.imshow(np.reshape(trainx[p_multimedia], (16, 16)))
    plt.subplot(3, 4, 10)
    plt.title('SERVICES')
    plt.imshow(np.reshape(trainx[p_service], (16, 16)))
    plt.subplot(3, 4, 11)
    plt.title('INTERACTIVE')
    plt.imshow(np.reshape(trainx[p_interactive], (16, 16)))
    plt.subplot(3, 4, 12)
    plt.title('GAMES')
    plt.imshow(np.reshape(trainx[p_games], (16, 16)))
    plt.show()

font_dict = {'family': 'Times New Roman',
         # 'style': 'italic',
         'weight': 'normal',
        # 'color':  'darkred',
        'size': 14,
        }
# 混淆矩阵
def plot_confusion_matrix(title, cm, labels_name=labels):
    # cm = confusion_matrix(test_y, np.argmax(pred_y, axis = 0))
    # cm = confusion_matrix(test_y, pred_y)  # > 0.5)
    print(cm)
    # labels_name = labels
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # print(cm)
    cm = cm.T
    plt.imshow(cm.T, interpolation='nearest', alpha=1.0)  # 在特定的窗口上显示图像

    plt.title(title, font_dict)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=30, fontsize=8.5, fontproperties = 'Times New Roman')  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, rotation=20, fontsize=10, fontproperties = 'Times New Roman')  # 将标签印在y轴坐标上
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            temp = cm[first_index][second_index]
            # if temp == 0.0 or temp == 100.0:
            #     plt.text(first_index, second_index, int(temp), va='center',
            #              ha='center',
            #              fontsize=13.5)
            # else:
            plt.text(first_index, second_index, r'{0:.2f}'.format(temp), va='center',
                     ha='center',
                     fontproperties = 'Times New Roman',
                     fontsize=12)
    plt.ylabel('True', font_dict)
    plt.xlabel('Predicted', font_dict)
    plt.savefig(get_path() + title + '.jpg', dpi=1200)
    plt.show()


# loss and acc
# 画指标折线图
def plt_index_of_model(epochs, loss, accuracy, val_loss, val_accuracy, title='Model'):
    x = np.arange(1, epochs + 1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs', font_dict)
    ax1.set_ylabel('Acc', font_dict)
    ax1.tick_params('y', colors='r')
    ax1.plot(x, accuracy, 'g+-', label='accuracy')
    ax1.plot(x, val_accuracy, 'c.-', label='val_accuracy')

    ax2 = ax1.twinx()
    # ax2.plot(x, accuracy, 'g', label='accuracy')
    # ax2.plot(x, val_accuracy, 'c', label='val_accuracy')
    # ax2.plot(x, loss, 'r', label='loss')
    # ax2.plot(x, val_loss, 'b', label='val_loss')
    ax1.legend(loc=(2/3,1/3))
    # ax1.grid()

    ax2.set_ylabel('Loss', font_dict)
    ax2.tick_params('y', colors='b')
    ax2.plot(x, loss, 'ro-', label='loss')
    ax2.plot(x, val_loss, 'b^-', label='val_loss')

    ax1.set_title(title, font_dict)

    ax2.legend(loc=(2/3,2/3))
    plt.savefig(get_path() + title + '.jpg', dpi=600)
    plt.show()

def plt_alpha(dim_set, alpha, mean_auroc, mean_fpr, title='Performance of different alpha'):
    """
    画VL方法
    :param dim_set:
    :param alpha:
    :param mean_auroc:
    :param val_loss:
    :param mean_fpr:
    :param title:
    :return:
    """
    x = dim_set
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # fig, ax1 = plt.subplots()

    ax1.set_xlabel('Dimension of principal subspace', font_dict)
    ax1.set_ylabel('Value', font_dict)
    ax1.tick_params('y', colors='r')
    ax1.plot(x, mean_auroc, 'g+-', label='mean_auroc')
    ax1.plot(x, mean_fpr, 'c.-', label='mean_fpr')

    ax2 = ax1.twinx()
    # ax2.plot(x, accuracy, 'g', label='accuracy')
    # ax2.plot(x, val_accuracy, 'c', label='val_accuracy')
    # ax2.plot(x, loss, 'r', label='loss')
    # ax2.plot(x, val_loss, 'b', label='val_loss')
    ax1.legend()
    # ax1.grid()

    ax2.set_ylabel('Alpha', font_dict)
    ax2.tick_params('y', colors='b')
    ax2.plot(x, alpha, 'ro-', label='alpha')

    ax1.set_title(title, font_dict)

    plt.legend()
    plt.savefig(get_path() + title + '.jpg', dpi=600)
    plt.show()

#画折线图
def plt_line(title, x_label, y_label, x, dict=None):
    if dict is None:
        print("折线图输入数据为空")
        return
    color = [
        ['r--', 'ro-'],
        ['g--', 'g+-'],
        ['b--', 'b^-'],
        ['c--', 'c.-'],
    ]
    i = 0
    fig = plt.figure(figsize=(7, 6))
    a = fig.add_subplot(111)
    # a = ax1.add_axes([0.1, 0.1, 0.8, 0.8])
    for key,value in dict.items():
        a.plot(x, value, color[i][0], label=key)
        a.plot(x, value, color[i][1],)
        i += 1
    # plt.plot(x, loss, 'ro-', x, accuracy, 'g+-', x, val_loss, 'b^-', x, val_accuracy, 'c.-')
    # a.set_ylim(0.98, 1.0)
    a.set_title(title, font_dict)
    a.set_xlabel(x_label, font_dict)
    a.set_ylabel(y_label, font_dict)
    plt.legend()
    plt.savefig(get_path() + title + '.jpg', dpi=600)
    plt.show()


#画柱状图
def plt_histogram(title, x_label, y_label, x, dict=None, threshold=14.321300293857709):
    if dict is None:
        print("折线图输入数据为空")
        return
    color = [
        ['r--', 'r'],
        ['g--', 'g'],
        ['b--', 'b'],
        ['c--', 'c'],
    ]
    i = 0
    # ax3 = plt.subplot(2, 1, 2)
    # plt.sca(ax3)
    # x1 = [1, 5, 9, 13, 17, 21, 25, 29]  # x轴点效率位置
    # x2 = [i + 1 for i in x1]  # x轴线效率位置
    # x3 = [i + 2 for i in x1]  # x轴面效率位置
    # y1 = [i[0] for i in rcount]  # y轴点效率位置
    # y2 = [i[1] for i in rcount]  # y轴线效率位置
    # y3 = [i[2] for i in rcount]  # y轴面效率位置
    width = 0.2
    x0 = list(range(len(x)))
    for key,value in dict.items():
        # plt.plot(x0, value, color[i][0], label=key)
        if key == 'Local Threshold':
            plt.plot(x0, value, color='b', linewidth=2, linestyle="--", label=key)
        else :
            plt.bar(x0, value, width=width, fc=color[i][1], tick_label=x, label=key)
            plt.xticks(rotation=23, fontsize=10, fontproperties = 'Times New Roman')
            i += 1
            x0 = [i1 + width for i1 in x0]
    line, = plt.plot(x0, [threshold] * len(x), color='c', linewidth=2, linestyle="--", label='Global Threshold')
    # plt.plot(x, loss, 'ro-', x, accuracy, 'g+-', x, val_loss, 'b^-', x, val_accuracy, 'c.-')
    plt.title(title, font_dict)
    plt.xlabel(x_label, font_dict)
    plt.ylabel(y_label, font_dict)
    plt.legend()
    plt.savefig(get_path() + title + '.jpg', dpi=600)
    plt.show()

def plt_metrics(title, x_label, y_label, x, dict=None,):
    if dict is None:
        print("折线图输入数据为空")
        return
    color = [
        ['r--', 'r'],
        ['g--', 'g'],
        ['b--', 'b'],
        ['c--', 'c'],
    ]
    i = 0
    width = 0.2
    x0 = list(range(len(x)))
    for key,value in dict.items():
        plt.bar(x0, value, width=width, fc=color[i][1], tick_label=x, label=key)
        plt.xticks(rotation=30, fontproperties = 'Times New Roman')
        plt.yticks(fontproperties = 'Times New Roman')
        i += 1
        x0 = [i1 + width for i1 in x0]
    # plt.plot(x, loss, 'ro-', x, accuracy, 'g+-', x, val_loss, 'b^-', x, val_accuracy, 'c.-')
    plt.title(title, font_dict)
    plt.xlabel(x_label, font_dict)
    plt.ylabel(y_label, font_dict)
    plt.legend()
    plt.savefig(get_path() + title + '.jpg', dpi=600)
    plt.show()

def plt_feat_importance(model, dim):
    features = read_features() # 248 维
    features += ["Placeholder1", "Placeholder2", "Placeholder3", "Placeholder4", "Placeholder5", "Placeholder6", "Placeholder7", "Placeholder8", ]
    importances = model.feature_importances_[:dim]
    indices = np.argsort(importances)[::-1]
    num_features = len(importances)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices], color="blue", align="center")
    plt.xticks(range(num_features), [features[i] for i in indices], rotation=30, fontproperties = 'Times New Roman')
    plt.xlim([-1, num_features])
    plt.show()