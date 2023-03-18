import numpy as np
from matplotlib import pyplot as plt
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
    model.save_weights('./output/model/' + name)


def load_model(model, name='', lis=None):
    if model is None:
        return
    if name == '':
        name = list_to_str(lis) + 'weights.h5'
    model.load_weights('./output/model/' + name)
    return model


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


# 混淆矩阵
def plot_confusion_matrix(title, test_y, pred_y, labels_name=labels):
    # cm = confusion_matrix(test_y, np.argmax(pred_y, axis = 0))
    cm = confusion_matrix(test_y, pred_y)  # > 0.5)
    print(cm)
    # labels_name = labels
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


# loss and acc
# 画指标折线图
def plt_index_of_model(epochs, loss, accuracy, val_loss, val_accuracy, title='Model'):
    x = np.arange(1, epochs + 1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Acc/%')
    ax1.tick_params('y', colors='r')
    ax1.plot(x, accuracy, 'g+-', label='accuracy')
    ax1.plot(x, val_accuracy, 'c.-', label='val_accuracy')

    ax2 = ax1.twinx()
    # ax2.plot(x, accuracy, 'g', label='accuracy')
    # ax2.plot(x, val_accuracy, 'c', label='val_accuracy')
    # ax2.plot(x, loss, 'r', label='loss')
    # ax2.plot(x, val_loss, 'b', label='val_loss')
    ax1.legend()
    # ax1.grid()

    ax2.set_ylabel('Loss')
    ax2.tick_params('y', colors='b')
    ax2.plot(x, loss, 'ro-', label='loss')
    ax2.plot(x, val_loss, 'b^-', label='val_loss')

    ax1.set_title(title + '- history')

    plt.legend()
    plt.savefig('./output/figure/{}.jpg'.format(title))
    plt.show()

def plot_twin(_y1, _y2, _ylabel1, _ylabel2):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('time')
    ax1.set_ylabel(_ylabel1, color=color)
    ax1.plot(_y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:red'
    ax2.set_ylabel(_ylabel2, color=color)
    ax2.plot(_y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
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
    for key,value in dict.items():
        plt.plot(x, value, color[i][0], label=key)
        plt.plot(x, value, color[i][1],)
        i += 1
    # plt.plot(x, loss, 'ro-', x, accuracy, 'g+-', x, val_loss, 'b^-', x, val_accuracy, 'c.-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
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
    plt.xticks(range(num_features), [features[i] for i in indices], rotation=30)
    plt.xlim([-1, num_features])
    plt.show()