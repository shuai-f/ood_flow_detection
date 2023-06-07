'''
Script to run baseline with cross-entropy loss on
'''
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from model import *
from util.ood_detection import VirtualLogit, LocalThreshold, get_ood_dict
from util.utils import list_to_str, plt_index_of_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_option():
    parser = argparse.ArgumentParser('arguments for training baseline DNN model')
    # training params
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size training'
                        )
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate training'
                        )
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--model', type=str, default='MLP',
                        help='DNN model to choose from ("MLP","RNN","LSTM","GRU")')

    # dataset params
    parser.add_argument('--data', type=str, default='moore',
                        help='Dataset to choose from ("moore", "self", "iscx")'
                        )
    parser.add_argument('--n_data_train', type=int, default=60000,
                        help='number of data points used for training both stage 1 and 2'
                        )
    # model architecture
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='output tensor dimension from projector'
                        )
    parser.add_argument('--activation', type=str, default='leaky_relu',
                        help='activation function between hidden layers'
                        )
    # output options
    parser.add_argument('--write_summary', action='store_true',
                        help='write summary for tensorboard'
                        )
    parser.add_argument('--draw_figures', action='store_true',
                        help='produce figures for the projections'
                        )

    args = parser.parse_args()
    return args

def get_splits(num=6):
    if num == 1:
        trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', ]
        oodLabels = ['MULTIMEDIA', 'SERVICES', 'INTERACTIVE', 'GAMES', ]
    elif num == 2:
        trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'SERVICES', 'DATABASE', 'FTP-DATA', ]
        oodLabels = ['MULTIMEDIA', 'P2P', 'INTERACTIVE', 'GAMES', ]
    # elif num == 3:
    #     trainLabels = ['WWW', 'MULTIMEDIA', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'SERVICES', 'DATABASE', 'FTP-DATA', ]
    #     oodLabels = ['MAIL', 'P2P', 'INTERACTIVE', 'GAMES', ]
    elif num == 4:
        trainLabels = ['WWW', 'MAIL', 'ATTACK', 'SERVICES', 'DATABASE', 'MULTIMEDIA', 'P2P', 'INTERACTIVE', ]
        oodLabels = ['FTP-CONTROL', 'FTP-PASV', 'FTP-DATA', 'GAMES', ]
    # elif num == 5:
    #     trainLabels = ['WWW', 'MULTIMEDIA', 'FTP-CONTROL', 'FTP-PASV', 'INTERACTIVE', 'P2P', 'SERVICES', 'FTP-DATA', ]
    #     oodLabels = ['MAIL', 'DATABASE', 'ATTACK', 'GAMES', ]
    elif num == 6:
        trainLabels = ['WWW', 'MULTIMEDIA', 'MAIL', 'FTP-PASV', 'INTERACTIVE', 'P2P', 'SERVICES', 'FTP-DATA', ]
        oodLabels = ['FTP-CONTROL', 'DATABASE', 'ATTACK', 'GAMES', ]

    return trainLabels, oodLabels

WEIGHT_PATH = './output/weight/'
model_name_suffix = '-basemodel.h5'

def main(model='CNN', batch_size=128, lr=0.001):
    args = parse_option()

    print(args)
    # args.model = model
    # args.batch_size = batch_size
    # args.lr = lr


    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # 0. Load data
    print('Loading {} data...'.format(args.data))
    # if args.data == 'mnist':
    #     mnist = tf.keras.datasets.mnist
    if args.data == 'moore':
        trainLabels, oodLabels = get_splits()
        input_npy_moore_dir = './inputs/npy/moore/'
        splits_dir = input_npy_moore_dir + list_to_str(oodLabels) + '_'
        train_x = np.load(splits_dir + 'train_x.npy')
        train_y = np.load(splits_dir + 'train_y.npy')
        test_x = np.load(splits_dir + 'test_x.npy')
        test_y = np.load(splits_dir + 'test_y.npy')
        ood_x = np.load(splits_dir + 'ood_x.npy')
        ood_y = np.load(splits_dir + 'ood_y.npy')
        # x_train, x_test = x_train / 25535.0, x_test / 25535.0
        num_pixel = 16 * 16
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
        print(train_x.shape, test_x.shape)
    elif args.data == 'self':
        from self.self_dataset import train_labels, ood_labels, read_data
        trainLabels = train_labels
        oodLabels = ood_labels
        train_x, train_y, test_x, test_y, ood_x, ood_y = read_data(oodLabels)
        num_pixel = 9 * 9
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
    elif args.data == 'iscx':
        from self.iscx import train_labels, ood_labels, read_data
        trainLabels = train_labels
        oodLabels = ood_labels
        train_x, train_y, test_x, test_y, ood_x, ood_y = read_data(oodLabels)
        num_pixel = 9 * 9
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
    else :
        print("Unknown Dataset Name :{}".format(args.data))
        return

    model_name = '{}_{}-bs_{}-lr_{}'.format(
        args.data, args.model, args.batch_size, args.lr)

    # 1. the baseline model
    if args.model == 'MLP':
        model = MLP(normalize=False, activation=args.activation)
        model.build(input_shape=[None, num_pixel, ])
    else :
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        ood_x = ood_x.reshape(-1, num_pixel, 1).astype(np.float32)
        if args.model == 'RNN':
            model = SimpleRNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'LSTM':
            model = LSTM(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'GRU':
            model = GRU(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'CNN':
            model = CNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'SimpleCNN':
            model = SimpleCNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        else :
            print("Unknown Model Name :{}".format(args.model))
            return
        # model.build(input_shape=[None, num_pixel, 1])
    cce_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    # simulate low data regime for training
    n_train = train_x.shape[0]
    shuffle_idx = np.arange(n_train)
    np.random.shuffle(shuffle_idx)

    train_x = train_x[shuffle_idx][:args.n_data_train]
    train_y = train_y[shuffle_idx][:args.n_data_train]
    print('Training dataset shapes after slicing:')
    print(train_x.shape, train_y.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(10000).batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).batch(args.batch_size)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_ACC')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_ACC')

    @tf.function
    def train_step_baseline(x, y):
        with tf.GradientTape() as tape:
            y_preds = model(x, training=True)
            loss = cce_loss_obj(y, y_preds)

        gradients = tape.gradient(loss,
                                  model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      model.trainable_variables))

        train_loss(loss)
        train_acc(y, y_preds)

    @tf.function
    def test_step_baseline(x, y):
        y_preds = model(x, training=False)
        t_loss = cce_loss_obj(y, y_preds)
        test_loss(t_loss)
        test_acc(y, y_preds)

    # model_name = 'baseline'
    if args.write_summary:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/%s/%s/%s/train' % (
            model_name, args.data, current_time)
        test_log_dir = 'logs/%s/%s/%s/test' % (
            model_name, args.data, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    acc = 0
    for epoch in range(args.epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        for x, y in train_ds:
            train_step_baseline(x, y)

        if args.write_summary:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

        for x_te, y_te in test_ds:
            test_step_baseline(x_te, y_te)

        if args.write_summary:
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_acc.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_acc.result() * 100,
                              test_loss.result(),
                              test_acc.result() * 100))
        if epoch == args.epoch - 1 :
            acc = test_acc.result() * 100
        history['loss'].append(train_loss.result())
        history['accuracy'].append(train_acc.result() * 100)
        history['val_loss'].append(test_loss.result())
        history['val_accuracy'].append(test_acc.result() * 100)
    with open('./output/base_model_output/{}_diff_args'.format(args.model), 'a') as f:
        f.write('\n{},{},{}'.format(args.batch_size, args.lr, acc))
    plt_index_of_model(args.epoch, history['loss'], history['accuracy'], history['val_loss'],history['val_accuracy'], title=args.model)
    # 2. Check learned embedding

    print(model.summary())

    w, b = model.output_layer.get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)
    model.save_weights(WEIGHT_PATH + model_name + model_name_suffix)
    # train_x_slices = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size)
    # train_x = None
    # train_y = None
    # for x, y in train_x_slices:
    #     x = model.get_last_hidden(x, training=False)
    #     x = x.numpy()
    #     # print("x {}, {}".format(x, x.shape))
    #     if train_x is None:
    #         train_x = x
    #         train_y = y
    #     else:
    #         train_x = np.concatenate((train_x, x), axis=0)
    #         train_y = np.concatenate((train_y, y), axis=0)
    # test_x = model.get_last_hidden(test_x).numpy()
    # ood_x = model.get_last_hidden(ood_x).numpy()
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, model, w, b, trainLabels, oodLabels)
    # ood_x = get_ood_dict(ood_x, ood_y, oodLabels)
    # VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, model, w, b, trainLabels, oodLabels)


def load_model():
    args = parse_option()
    print(args)

    print('Loading {} data...'.format(args.data))
    if args.data == 'moore':
        trainLabels, oodLabels = get_splits()
        input_npy_moore_dir = './inputs/npy/moore/'
        splits_dir = input_npy_moore_dir + list_to_str(oodLabels) + '_'
        train_x = np.load(splits_dir + 'train_x.npy')
        train_y = np.load(splits_dir + 'train_y.npy')
        test_x = np.load(splits_dir + 'test_x.npy')
        test_y = np.load(splits_dir + 'test_y.npy')
        ood_x = np.load(splits_dir + 'ood_x.npy')
        ood_y = np.load(splits_dir + 'ood_y.npy')
        # x_train, x_test = x_train / 25535.0, x_test / 25535.0
        num_pixel = 16 * 16
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
        ood_x = ood_x.reshape(-1, num_pixel).astype(np.float32)
        print(train_x.shape, test_x.shape)
    elif args.data == 'self':
        from self.self_dataset import train_labels, ood_labels, read_data
        trainLabels = train_labels
        oodLabels = ood_labels
        train_x, train_y, test_x, test_y, ood_x, ood_y = read_data(oodLabels)
        num_pixel = 9 * 9
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
    elif args.data == 'iscx':
        from self.iscx import train_labels, ood_labels, read_data
        trainLabels = train_labels
        oodLabels = ood_labels
        train_x, train_y, test_x, test_y, ood_x, ood_y = read_data(oodLabels)
        num_pixel = 9 * 9
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
    else:
        print("Unknown Dataset Name :{}".format(args.data))
        return

    model_name = '{}_{}-bs_{}-lr_{}'.format(
        args.data, args.model, args.batch_size, args.lr)
    print(model_name)

    if args.model == 'MLP':
        model = MLP(normalize=False, activation=args.activation)
        model.build(input_shape=[None, num_pixel, ])
    else :
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        ood_x = ood_x.reshape(-1, num_pixel, 1).astype(np.float32)
        if args.model == 'RNN':
            model = SimpleRNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'LSTM':
            model = LSTM(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'GRU':
            model = GRU(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'CNN':
            model = CNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'SimpleCNN':
            model = SimpleCNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        else :
            print("Unknown Model Name :{}".format(args.model))
            return
        model.build(input_shape=[None, num_pixel, 1])

    model.load_weights(WEIGHT_PATH + model_name + model_name_suffix)

    if args.draw_figures:
        # get the projections from the last hidden layer before output
        # projecting data with the trained encoder, projector
        x_tr_proj = None
        # 数据集过大需要切片
        train_x_slices = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size)
        for x, y in train_x_slices:
            if x_tr_proj is None:
                x_tr_proj = model.get_last_hidden(x)
            else:
                x_tr_proj = np.concatenate((x_tr_proj, model.get_last_hidden(x)), axis=0)
        x_te_proj = model.get_last_hidden(test_x)
        # convert tensor to np.array
        x_te_proj = x_te_proj.numpy()
        print(x_tr_proj.shape, x_te_proj.shape)
        # do PCA for the projected data
        pca = PCA(n_components=2)
        pca.fit(x_tr_proj)
        x_te_proj_pca = pca.transform(x_te_proj)

        x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
        x_te_proj_pca_df['label'] = test_y
        # PCA scatter plot
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x='PC1', y='PC2',
                             data=x_te_proj_pca_df,
                             palette='tab10',
                             hue='label',
                             linewidth=0,
                             alpha=0.6,
                             ax=ax
                             )

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        title = 'Data: %s; Embedding: %s' % (args.data, args.model)
        ax.set_title(title)
        fig.savefig('figs/PCA_plot_%s_%s_last_layer.png' % (args.data, args.model))
        # density plot for PCA
        g = sns.jointplot(x='PC1', y='PC2', data=x_te_proj_pca_df,
                          kind="hex"
                          )
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
        g.savefig('figs/Joint_PCA_plot_%s_%s_last_layer.png' % (args.data, args.model))

        # check learned embedding using LDA : 测试集对比
        pca = PCA(n_components=3)  # LinearDiscriminantAnalysis(n_components=3)
        pca.fit(x_tr_proj, train_y)
        x_te_proj_lda = pca.transform(x_te_proj)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.title("PCA-{}".format(args.model))
        ax.scatter(x_te_proj_lda[:, 0], x_te_proj_lda[:, 1], x_te_proj_lda[:, 2], c=test_y)
        plt.savefig("figs/PCA-origin-{}.png".format(args.model), dpi=600)
        plt.show()

        pca = PCA(n_components=3)
        pca.fit(train_x.reshape(-1, num_pixel).astype(np.float32), train_y)
        x_te_lda = pca.transform(test_x.reshape(-1, num_pixel).astype(np.float32))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.title("PCA-{}".format('origin'))
        ax.scatter(x_te_lda[:, 0], x_te_lda[:, 1], x_te_lda[:, 2], c=test_y)
        plt.savefig("figs/PCA-origin-{}.png".format('origin'), dpi=600)
        plt.show()

    print(model.summary())
    w, b = model.output_layer.get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)
    # train_x = model.get_last_hidden(train_x).numpy()
    train_x_slices = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size)
    train_x = None
    train_y = None
    for x, y in train_x_slices:
        x = model.get_last_hidden(x, training=False)
        # print("x {}, {}".format(x, x.shape))
        if train_x is None:
            train_x = x
            train_y = y
        else:
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)
    test_x = model.get_last_hidden(test_x).numpy()
    ood_x = model.get_last_hidden(ood_x).numpy()
    from util.ood_func import vl_score, msp_score
    softmax = model.output_layer
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, model.output_layer, w, b,
    #                trainLabels, oodLabels, vl_score, DIM=25)
    from util.ood_func import energy_score, msp_score, mahalanobis_score, maxlogit_socre, odin_score, residual_score, \
        nusa_score, vl_score

    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
    #                , w, b, trainLabels, oodLabels, msp_score)  # mahalanobis_score)
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
    #                , w, b, trainLabels, oodLabels, maxlogit_socre)  # mahalanobis_score)
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
    #                , w, b, trainLabels, oodLabels, nusa_score)  # mahalanobis_score)
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
    #                , w, b, trainLabels, oodLabels, energy_score)  # mahalanobis_score)
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
    #                , w, b, trainLabels, oodLabels, residual_score)  # mahalanobis_score)
    # LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
    #                , w, b, trainLabels, oodLabels, mahalanobis_score)
    LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
                   , w, b, trainLabels, oodLabels, vl_score, DIM=85)  # mahalanobis_score)

    ood_x = get_ood_dict(ood_x, ood_y, oodLabels)
    VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, model, w, b, trainLabels, oodLabels)


if __name__ == '__main__':
    # main()
    load_model()
    # for model in ['MLP']:
    #     for bs in [128,64,32]:
    #         for lr in [0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095]:#[0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.005, 0.01]:
    #             main(model, bs, lr)
