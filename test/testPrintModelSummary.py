import util.utils
from contrast_learning.model import CNN, SimpleRNN, MLP, GRU
import tensorflow as tf

def print_summary():
    num_pixel = 16 * 16
    model = MLP(normalize=False, activation='relu')
    model.build(input_shape=(None, num_pixel, ))
    model.call(tf.keras.layers.Input(shape=(num_pixel)))
    model.summary(show_trainable=True)
    model = SimpleRNN(normalize=False, activation='relu', input_shape=(num_pixel, 1))
    model.build(input_shape=(None, num_pixel, 1))
    model.call(tf.keras.layers.Input(shape=(num_pixel, 1)))
    model.summary(show_trainable=True)
    # model = LSTM(normalize=False, activation='relu', input_shape=(num_pixel, 1))
    # model.build(input_shape=(None, num_pixel, 1))
    # model.call(tf.keras.layers.Input(shape=(num_pixel, 1)))
    # model.summary(show_trainable=True)
    model = GRU(normalize=False, activation='relu', input_shape=(num_pixel, 1))
    model.build(input_shape=(None, num_pixel, 1))
    model.call(tf.keras.layers.Input(shape=(num_pixel, 1)))
    model.summary(show_trainable=True)
    model = CNN(normalize=False, activation='relu', input_shape=(num_pixel, 1))
    model.build(input_shape=(None, num_pixel, 1))
    model.call(tf.keras.layers.Input(shape=(num_pixel, 1)))
    model.summary(show_trainable=True)

def plot_MLP():
    model = 'MLP'
    with open('./output/base_model_output/{}_diff_args'.format(model),'r') as f:
        lines = f.readlines()
    dict = {'batch_size = 32':[], 'batch_size = 64':[], 'batch_size = 128':[]}
    for i in range(62, len(lines)):
        line = lines[i]
        splits = line.split(',')

        dict['batch_size = {}'.format(splits[0])].append(float(splits[-1]) / 100)
    lr = [0.0001, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    util.utils.plt_line(model, 'Learning rate', 'Accuracy', lr, dict)

def plot_CNN():
    model = 'CNN'
    with open('./output/base_model_output/{}_diff_args'.format(model),'r') as f:
        lines = f.readlines()
    dict = {'batch_size = 32':[], 'batch_size = 64':[], 'batch_size = 128':[]}
    for i in range(0, len(lines)):
        line = lines[i]
        splits = line.split(',')

        dict['batch_size = {}'.format(splits[0])].append(float(splits[-1]) / 100)
    lr = [0.0001, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    # lr = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    util.utils.plt_line(model, 'Learning rate', 'Accuracy', lr, dict)

def plot_cl():
    model = 'MLP'
    with open('./output/CL/{}_diff_args'.format(model),'r') as f:
        lines = f.readlines()
    dict = {'batch_size = 32':[], 'batch_size = 64':[], 'batch_size = 128':[], 'batch_size = 512':[],}
    for line in lines:
        splits = line.split(',')
        dict['batch_size = {}'.format(splits[0])].append(float(splits[-2]))
    lr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    lr = [0.001, 0.0025, 0.004, 0.0055, 0.007, 0.0085, 0.01]
    util.utils.plt_line('Encoder : {}, loss : Supervised NT-Xent Loss, Temperature : 0.07 '.format(model), 'Learning rate', 'Accuracy', lr, dict)

def plot_metrics():
    with open('./output/CL/diff_metrics.csv', 'r') as f:
        lines = f.readlines()
    metrics = {'F1-Score':[], 'Precision':[], 'Recall':[], } #'Accuracy':[],
    for i in range(97, 97 + 14):#(62, 62 + 14):#(127, 127 + 14):#
        splits = lines[i].split(',')
        if 'residual' in splits[0]:
            continue
        if i % 2 == 1:
            i = 0
            j = 0 + 4 + 4
            metrics['F1-Score'].append(float(splits[j+1]))
            metrics['Precision'].append(float(splits[j+2]))
            metrics['Recall'].append(float(splits[j+3]))
            # metrics['Accuracy'].append(float(splits[j+4]))
    print(metrics)
    x = ['MSP','MaxLogit','ODIN','Energy','Mahalanobis', "VirtualLogit"]
    # util.utils.plt_line('Metrics', 'Method', 'Value/%', x, metrics)
    util.utils.plt_metrics('Metrics', 'Method', 'Value/%', x, metrics)

def plot_threshold():
    with open('./output/CL/local threshold1.csv', 'r') as f:
        thres, seen, unseen = f.readline().split(',')
        lines = f.readlines()
    dict = {'Local Threshold':[], 'In-distribution':[], 'Out-of-Distribution':[]}
    for line in lines:
        splits = line.split(',')
        dict['Local Threshold'].append(float(splits[0]))
        dict['In-distribution'].append(abs(float(splits[1])))
        dict['Out-of-Distribution'].append(float(splits[2]))
    print(thres, seen, unseen)
    # trainLabels = ['WWW', 'MULTIMEDIA', 'MAIL', 'FTP-PASV', 'INTERACTIVE', 'P2P', 'SERVICES', 'FTP-DATA', ]
    trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'SERVICES', 'DATABASE', 'FTP-DATA', ]
    util.utils.plt_histogram('Average OOD Score', 'Classification', 'Score', trainLabels, dict)

def plot_history():
    model = 'RNN'
    with open('./output/base_model_output/{}'.format(model),'r') as f:
        lines = f.readlines()
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    first = 13
    for i in range(first, first + 50):
        splits = lines[i].split(',')
        history['loss'].append(float(splits[1].split(':')[1]))
        history['accuracy'].append(float(splits[2].split(':')[1]) / 100)
        history['val_loss'].append(float(splits[3].split(':')[1]))
        history['val_accuracy'].append(float(splits[4].split(':')[1]) / 100)
    util.utils.plt_index_of_model(50, history['loss'], history['accuracy'], history['val_loss'],
                           history['val_accuracy'], title=model)

if __name__ == '__main__':
    # print_summary()
    # plot_MLP()
    # plot_CNN()
    # plot_cl()
    # plot_threshold()
    # plot_metrics()
    plot_history()