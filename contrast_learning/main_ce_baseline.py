'''
Script to run baseline with cross-entropy loss on
'''
import argparse
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from contrast_learning.model import *
from self import self_dataset
from ood_detection import VirtualLogit, LocalThreshold, get_ood_dict
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
                        help='Dataset to choose from ("mnist", "moore", "self")'
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

WEIGHT_PATH = './output/weight/'
def main():
    args = parse_option()
    print(args)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # 0. Load data
    print('Loading {} data...'.format(args.data))
    # if args.data == 'mnist':
    #     mnist = tf.keras.datasets.mnist
    if args.data == 'moore':
        trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', ]
        oodLabels = ['MULTIMEDIA', 'P2P', 'INTERACTIVE', 'GAMES', ]
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
        from self.self_dataset import train_labels, ood_labels
        trainLabels = train_labels
        oodLabels = ood_labels
        train_x, train_y, test_x, test_y, ood_x, ood_y = self_dataset.read_data(oodLabels)
        num_pixel = 9 * 9
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
    else :
        print("Unknown Dataset Name :{}".format(args.data))
        return

    model_name = '{}_{}-bs_{}-lr_{}'.format(
        args.data, args.model, args.batch_size, args.lr)

    # 1. the baseline MLP model
    if args.model == 'MLP':
        model = MLP(normalize=False, activation=args.activation)
    elif args.model == 'RNN':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = SimpleRNN(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    elif args.model == 'LSTM':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = LSTM(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    elif args.model == 'GRU':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = GRU(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    elif args.model == 'CNN':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = CNN(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    else :
        print("Unknown Model Name :{}".format(args.model))
        return
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
        (train_x, train_y)).shuffle(50000).batch(args.batch_size)

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
        history['loss'].append(train_loss.result())
        history['accuracy'].append(train_acc.result() * 100)
        history['val_loss'].append(test_loss.result())
        history['val_accuracy'].append(test_acc.result() * 100)
    plt_index_of_model(args.epoch, history['loss'], history['accuracy'], history['val_loss'],history['val_accuracy'], title=args.model)
    # get the projections from the last hidden layer before output
    x_tr_proj = model.get_last_hidden(train_x)
    x_te_proj = model.get_last_hidden(test_x)
    # convert tensor to np.array
    x_tr_proj = x_tr_proj.numpy()
    x_te_proj = x_te_proj.numpy()
    print(x_tr_proj.shape, x_te_proj.shape)
    # 2. Check learned embedding
    if args.draw_figures:
        # do PCA for the projected data
        pca = PCA(n_components=2)
        pca.fit(x_tr_proj)
        x_te_proj_pca = pca.transform(x_te_proj)

        x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
        x_te_proj_pca_df['label'] = test_y
        # PCA scatter plot
        fig, ax = plt.subplots()
        ax = sns.scatterplot('PC1', 'PC2',
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
        title = 'Data: %s; Embedding: MLP' % args.data
        ax.set_title(title)
        fig.savefig('figs/PCA_plot_%s_MLP_last_layer.png' % args.data)
        # density plot for PCA
        g = sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df,
                          kind="hex"
                          )
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
        g.savefig('figs/Joint_PCA_plot_%s_MLP_last_layer.png' % args.data)

    print(model.summary())

    w, b = model.output_layer.get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)
    model.save_weights(WEIGHT_PATH + model_name + '-basemodel.h5')
    train_x = model.get_last_hidden(train_x).numpy()
    test_x = model.get_last_hidden(test_x).numpy()
    ood_x = model.get_last_hidden(ood_x).numpy()
    LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, model, w, b, trainLabels, oodLabels)
    ood_x = get_ood_dict(ood_x, ood_y, oodLabels)
    # VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, model, w, b, trainLabels, oodLabels)


def load_model():
    args = parse_option()
    print(args)

    print('Loading {} data...'.format(args.data))
    trainLabels = ['WWW', 'MAIL', 'FTP-CONTROL', 'FTP-PASV', 'ATTACK', 'P2P', 'DATABASE', 'FTP-DATA', ]
    oodLabels = ['MULTIMEDIA', 'P2P', 'INTERACTIVE', 'GAMES', ]
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
    model_name = '{}_{}-bs_{}-lr_{}'.format(
        args.data, args.model, args.batch_size, args.lr)

    if args.model == 'MLP':
        model = MLP(normalize=False, activation=args.activation)
        model.build(input_shape=[None, 16 * 16, ])
    elif args.model == 'RNN':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = SimpleRNN(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    elif args.model == 'LSTM':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = LSTM(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    elif args.model == 'GRU':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = GRU(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    elif args.model == 'CNN':
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        model = CNN(normalize=False, activation=args.activation, input_shape=(16*16, 1))
    else :
        print("Unknown Model Name :{}".format(args.model))
        return

    model.load_weights(WEIGHT_PATH + model_name + '-basemodel.h5')

    print(model.summary())
    w, b = model.output_layer.get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)
    train_x = model.get_last_hidden(train_x).numpy()
    test_x = model.get_last_hidden(test_x).numpy()
    ood_x = model.get_last_hidden(ood_x).numpy()
    LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, model.output_layer, w, b, trainLabels, oodLabels)

    ood_x = get_ood_dict(ood_x, ood_y, oodLabels)
    VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, model, w, b, trainLabels, oodLabels)


if __name__ == '__main__':
    # main()
    load_model()
