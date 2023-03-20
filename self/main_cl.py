'''
Script to run various two-stage supervised contrastive loss functions on
'''
import argparse
import datetime
import numpy as np
import tensorflow_addons as tfa
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from contrast_learning.model import *
import contrast_learning.losses
from main import self_dataset
from util.ood_detection import VirtualLogit, get_ood_dict, LocalThreshold
from util.utils import list_to_str, plt_index_of_model, plt_line

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOSS_NAMES = {
    'max_margin': 'Max margin contrastive',
    'npairs': 'Multiclass N-pairs',
    'sup_nt_xent': 'Supervised NT-Xent',
    'triplet-hard': 'Triplet hard',
    'triplet-semihard': 'Triplet semihard',
    'triplet-soft': 'Triplet soft'
}

WEIGHT_PATH = './output/weight/'

def parse_option():
    parser = argparse.ArgumentParser('arguments for two-stage training ')
    # training params
    parser.add_argument('--batch_size_1', type=int, default=512,
                        help='batch size for stage 1 pretraining'
                        )
    parser.add_argument('--batch_size_2', type=int, default=128,
                        help='batch size for stage 2 training'
                        )
    parser.add_argument('--lr_1', type=float, default=0.5,
                        help='learning rate for stage 1 pretraining'
                        )
    parser.add_argument('--lr_2', type=float, default=0.001,
                        help='learning rate for stage 2 training'
                        )
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs for training in stage1, the same number of epochs will be applied on stage2')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use, choose from ("adam", "lars", "sgd")'
                        )
    # loss functions
    parser.add_argument('--loss', type=str, default='sup_nt_xent',
                        help='Loss function used for stage 1, choose from ("max_margin", "npairs", "sup_nt_xent", "triplet-hard", "triplet-semihard", "triplet-soft")')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='margin for tfa.losses.contrastive_loss. will only be used when --loss=max_margin')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='distance metrics for tfa.losses.contrastive_loss, choose from ("euclidean", "cosine"). will only be used when --loss=max_margin')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='temperature for sup_nt_xent loss. will only be used when --loss=sup_nt_xent')
    parser.add_argument('--base_temperature', type=float, default=0.07,
                        help='base_temperature for sup_nt_xent loss. will only be used when --loss=sup_nt_xent')
    # dataset params
    parser.add_argument('--data', type=str, default='moore',
                        help='Dataset to choose from ("mnist", "moore", "self")'
                        )
    parser.add_argument('--n_data_train', type=int, default=60000,
                        help='number of data points used for training both stage 1 and 2'
                        )
    parser.add_argument('--model', type=str, default='MLP',
                        help='DNN model to choose from ("MLP","RNN","LSTM","GRU","CNN")')

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


def main():
    args = parse_option()
    # import resourcesoft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(args)

    # check args
    if args.loss not in LOSS_NAMES:
        raise ValueError('Unsupported loss function type {}'.format(args.loss))

    if args.optimizer == 'adam':
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=args.lr_1)
    elif args.optimizer == 'lars':
        from contrast_learning.lars_optimizer import LARSOptimizer
        # not compatible with tf2
        optimizer1 = LARSOptimizer(args.lr_1,
                                   exclude_from_weight_decay=['batch_normalization', 'bias'])
    elif args.optimizer == 'sgd':
        optimizer1 = tfa.optimizers.SGDW(learning_rate=args.lr_1,
                                         momentum=0.9,
                                         weight_decay=1e-4
                                         )
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=args.lr_2)

    model_name = '{}_{}-bs_{}-lr_{}_pre_to_cl'.format(
        args.loss, args.model, args.batch_size_1, args.lr_2)

    # 0. Load data

    print('Loading {} data...'.format(args.data))
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
        print(train_x.shape, test_x.shape)
    else :
        print("Unknown Dataset Name :{}".format(args.data))
        return

    pre_model_name = '{}_{}-bs_{}-lr_{}'.format(
        args.data, args.model, args.batch_size_2, args.lr_2)

    if args.model == 'MLP':
        encoder = MLP(normalize=False, activation=args.activation)
        # encoder.build(input_shape=[None, num_pixel, ])
    else :
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        ood_x = ood_x.reshape(-1, num_pixel, 1).astype(np.float32)
        if args.model == 'RNN':
            encoder = SimpleRNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'LSTM':
            encoder = LSTM(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'GRU':
            encoder = GRU(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'CNN':
            encoder = EncoderCNN(normalize=True, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'SimpleCNN':
            encoder = SimpleCNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        else :
            print("Unknown Model Name :{}".format(args.model))
            return
        # encoder.build(input_shape=[None, num_pixel, 1])
        # simulate low data regime for training
    n_train = train_x.shape[0]
    shuffle_idx = np.arange(n_train)
    np.random.shuffle(shuffle_idx)

    train_x = train_x[shuffle_idx][:args.n_data_train]
    train_y = train_y[shuffle_idx][:args.n_data_train]
    print('Training dataset shapes after slicing:')
    print(train_x.shape, train_y.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(5000).batch(args.batch_size_1)

    train_ds2 = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(10000).batch(args.batch_size_2)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).batch(args.batch_size_1)

    # 1. Stage 1: train encoder with multiclass N-pair loss
    # encoder = Encoder(normalize=True, activation=args.activation)
    # print("load model {}...".format(pre_model_name))
    # encoder.load_weights(WEIGHT_PATH + pre_model_name + "-basemodel.h5")
    projector = Projector(args.projection_dim,
                          normalize=True, activation=args.activation)

    if args.loss == 'max_margin':
        def loss_func(z, y): return contrast_learning.losses.max_margin_contrastive_loss(
            z, y, margin=args.margin, metric=args.metric)
    elif args.loss == 'npairs':
        loss_func = contrast_learning.losses.multiclass_npairs_loss
    elif args.loss == 'sup_nt_xent':
        def loss_func(z, y): return contrast_learning.losses.supervised_nt_xent_loss(
            z, y, temperature=args.temperature, base_temperature=args.base_temperature)
    elif args.loss.startswith('triplet'):
        triplet_kind = args.loss.split('-')[1]
        def loss_func(z, y): return contrast_learning.losses.triplet_loss(
            z, y, kind=triplet_kind, margin=args.margin)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # tf.config.experimental_run_functions_eagerly(True)
    @tf.function
    # train step for the contrastive loss
    def train_step_stage1(x, y):
        '''
        x: data tensor, shape: (batch_size, data_dim)
        y: data labels, shape: (batch_size, )
        '''
        with tf.GradientTape() as tape:
            # r = encoder.get_last_hidden(x, training=True)
            r = encoder(x, training=True)
            z = projector(r, training=True)
            loss = loss_func(z, y)

        gradients = tape.gradient(loss,
                                  encoder.trainable_variables + projector.trainable_variables)
        optimizer1.apply_gradients(zip(gradients,
                                       encoder.trainable_variables + projector.trainable_variables))
        train_loss(loss)

    @tf.function
    def test_step_stage1(x, y):
        # r = encoder.get_last_hidden(x, training=False)
        r = encoder(x, training=False)
        z = projector(r, training=False)
        t_loss = loss_func(z, y)
        test_loss(t_loss)

    print('Stage 1 training ...')
    loss_his = {'con_loss':[]}
    for epoch in range(args.epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for x, y in train_ds:
            train_step_stage1(x, y)

        for x_te, y_te in test_ds:
            test_step_stage1(x_te, y_te)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              test_loss.result()))
        loss_his['con_loss'].append(train_loss.result())
    plt_line('{} Contrast Loss'.format(args.model),'Epoch','Value',[i for i in range(args.epoch)],loss_his)

    # Stage 2: freeze the learned representations and then learn a classifier
    # on a linear layer using a softmax loss
    softmax = SoftmaxPred(len(trainLabels))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_ACC')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_ACC')

    cce_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    @tf.function
    # train step for the 2nd stage
    def train_step(x, y):
        '''
        x: data tensor, shape: (batch_size, data_dim)
        y: data labels, shape: (batch_size, )
        '''
        with tf.GradientTape() as tape:
            # r = encoder.get_last_hidden(x, training=False)
            r = encoder(x, training=False)
            y_preds = softmax(r, training=True)
            # y_preds = encoder(x, training=True)
            loss = cce_loss_obj(y, y_preds)

        # freeze the encoder, only train the softmax layer
        gradients = tape.gradient(loss,
                                  softmax.trainable_variables)
        optimizer2.apply_gradients(zip(gradients,
                                       softmax.trainable_variables))
        train_loss(loss)
        train_acc(y, y_preds)

    @tf.function
    def test_step(x, y):
        # r = encoder.get_last_hidden(x, training=False)
        r = encoder(x, training=False)
        y_preds = softmax(r, training=False)
        # y_preds = encoder(x, training=False)
        t_loss = cce_loss_obj(y, y_preds)
        test_loss(t_loss)
        test_acc(y, y_preds)

    if args.write_summary:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/{}/{}/{}/train'.format(
            model_name, args.data, current_time)
        test_log_dir = 'logs/{}/{}/{}/test'.format(
            model_name, args.data, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    print('Stage 2 training ...')
    history = {'loss':[], 'accuracy':[],'val_loss':[], 'val_accuracy':[]}
    for epoch in range(args.epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        for x, y in train_ds2:
            train_step(x, y)

        if args.write_summary:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

        for x_te, y_te in test_ds:
            test_step(x_te, y_te)

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
        history['accuracy'].append(train_acc.result())
        history['val_loss'].append(test_loss.result())
        history['val_accuracy'].append(test_acc.result())
    plt_index_of_model(args.epoch, history['loss'], history['accuracy'], history['val_loss'],history['val_accuracy'], title="{} CL".format(args.model))

    print(encoder.summary())
    encoder.save_weights(WEIGHT_PATH + model_name + '-encoder.h5')
    projector.save_weights(WEIGHT_PATH + model_name + '-projector.h5')
    softmax.save_weights(WEIGHT_PATH + model_name + '-softmax.h5')
    w, b = softmax.dense.get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)
    # train_x = encoder(train_x).numpy()
    train_x_slices = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size_2)
    train_x = None
    train_y = None
    for x, y in train_x_slices:
        x = encoder(x, training=False)
        # print("x {}, {}".format(x, x.shape))
        if train_x is None:
            train_x = x
            train_y = y
        else:
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)

    test_x = encoder(test_x).numpy()
    ood_x = encoder(ood_x).numpy()
    LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax, w, b, trainLabels, oodLabels)
    ood_x = get_ood_dict(ood_x, ood_y, oodLabels)
    VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax, w, b, trainLabels, oodLabels)


    if args.draw_figures:
        draw_figures(args, train_x, train_y, test_x, test_y, encoder, projector, num_pixel, model_name)


def load_model():
    args = parse_option()
    print(args)
    model_name = '{}_{}-bs_{}-lr_{}_pre_to_cl'.format(
        args.loss, args.model, args.batch_size_1, args.lr_2)

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
        print(train_x.shape, test_x.shape)
    else :
        print("Unknown Dataset Name :{}".format(args.data))
        return

    if args.model == 'MLP':
        encoder = MLP(normalize=False, activation=args.activation)
        encoder.build(input_shape=[None, num_pixel, ])
    else :
        train_x = train_x.reshape(-1, num_pixel, 1).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel, 1).astype(np.float32)
        ood_x = ood_x.reshape(-1, num_pixel, 1).astype(np.float32)
        if args.model == 'RNN':
            encoder = SimpleRNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'CNN':
            encoder = EncoderCNN(normalize=True, activation=args.activation, input_shape=(num_pixel, 1))
        elif args.model == 'SimpleCNN':
            encoder = SimpleCNN(normalize=False, activation=args.activation, input_shape=(num_pixel, 1))
        else :
            print("Unknown Model Name :{}".format(args.model))
            return
        encoder.build(input_shape=(None, num_pixel, 1))
    # encoder = Encoder(normalize=True, activation=args.activation)
    projector = Projector(args.projection_dim,
                          normalize=True, activation=args.activation)
    projector.build(input_shape=(None, args.projection_dim, ))
    softmax = SoftmaxPred(len(trainLabels))
    softmax.build(input_shape=(None, args.projection_dim, ))
    print(encoder.summary())

    encoder.load_weights(WEIGHT_PATH + model_name + '-encoder.h5')
    projector.load_weights(WEIGHT_PATH + model_name + '-projector.h5')
    softmax.load_weights(WEIGHT_PATH + model_name + '-softmax.h5')


    if args.draw_figures:
        draw_figures(args, train_x, train_y, test_x, test_y, encoder, projector, num_pixel, model_name)
    # performance
    w, b = softmax.dense.get_weights()
    w = w.T
    print(w, w.shape)
    print(b, b.shape)
    # train_x = encoder(train_x).numpy()
    save = False
    if save:
        train_x_slices = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size_2)
        train_x = None
        train_y = None
        for x, y in train_x_slices:
            x = encoder(x, training=False)
            # print("x {}, {}".format(x, x.shape))
            if train_x is None:
                train_x = x
                train_y = y
            else:
                train_x = np.concatenate((train_x, x), axis=0)
                train_y = np.concatenate((train_y, y), axis=0)
        test_x = encoder(test_x).numpy()
        np.save('./output/mid/train_x_cl.npy', train_x)
        np.save('./output/mid/train_y_cl.npy', train_y)
        np.save('./output/mid/test_x_cl.npy', test_x)
    else :
        train_x = np.load('./output/mid/train_x_cl.npy')
        train_y = np.load('./output/mid/train_y_cl.npy')
        test_x = np.load('./output/mid/test_x_cl.npy')
    ood_x = encoder(ood_x).numpy()
    from util.ood_func import energy_score, msp_score, mahalanobis_score, maxlogit_socre, odin_score , residual_score , nusa_score
    LocalThreshold(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax
                   , w, b, trainLabels, oodLabels, energy_score)
    # ood_x = get_ood_dict(ood_x, ood_y, oodLabels)
    # VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, softmax, w, b, trainLabels, oodLabels)

def draw_figures(args, train_x, train_y, test_x, test_y, encoder, projector, num_pixel, model_name):
    # projecting data with the trained encoder, projector
    x_tr_proj = None
    # 数据集过大需要切片
    train_x_slices = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size_2)
    for x, y in train_x_slices:
        if x_tr_proj is None:
            x_tr_proj = projector(encoder(x))
        else:
            x_tr_proj = np.concatenate((x_tr_proj, projector(encoder(x))), axis=0)
    x_te_proj = projector(encoder(test_x))
    # convert tensor to np.array
    # x_tr_proj = x_tr_proj.numpy()
    x_te_proj = x_te_proj.numpy()
    print(x_tr_proj.shape, x_te_proj.shape)

    # check learned embedding using PCA
    pca = PCA(n_components=2)
    pca.fit(x_tr_proj)
    x_te_proj_pca = pca.transform(x_te_proj)

    plt.figure(figsize=(4, 4))
    plt.plot(122)
    plt.title("PCA-{}".format(args.loss))
    plt.scatter(x_te_proj_pca[:, 0], x_te_proj_pca[:, 1], c=test_y)
    # plt.savefig("PCA-{}.png".format(args.loss), dpi=600)
    plt.show()

    # check learned embedding using LDA : 测试集对比
    pca = PCA(n_components=3)  # LinearDiscriminantAnalysis(n_components=3)
    pca.fit(x_tr_proj, train_y)
    x_te_proj_lda = pca.transform(x_te_proj)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("PCA-{}".format(args.loss))
    ax.scatter(x_te_proj_lda[:, 0], x_te_proj_lda[:, 1], x_te_proj_lda[:, 2], c=test_y)
    plt.savefig("figs/PCA-{}.png".format(args.loss), dpi=600)
    plt.show()

    pca = PCA(n_components=3)
    pca.fit(train_x.reshape(-1, num_pixel).astype(np.float32), train_y)
    x_te_lda = pca.transform(test_x.reshape(-1, num_pixel).astype(np.float32))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("PCA-{}".format('origin'))
    ax.scatter(x_te_lda[:, 0], x_te_lda[:, 1], x_te_lda[:, 2], c=test_y)
    plt.savefig("figs/PCA-{}.png".format('origin'), dpi=600)
    plt.show()

    x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
    x_te_proj_pca_df['label'] = test_y
    # PCA scatter plot
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x='PC1',
                         y='PC2',
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
    title = 'Data: {}\nEmbedding: {}\nbatch size: {}; LR: {}'.format(
        args.data, LOSS_NAMES[args.loss], args.batch_size_1, args.lr_1)
    ax.set_title(title)
    fig.savefig(
        'figs/PCA_plot_{}_{}_embed.png'.format(args.data, model_name))

    # density plot for PCA
    g = sns.jointplot(x='PC1', y='PC2', data=x_te_proj_pca_df,
                      kind="hex"
                      )
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle(title)

    g.savefig(
        'figs/Joint_PCA_plot_{}_{}_embed.png'.format(args.data, model_name))

if __name__ == '__main__':
    # main()
    load_model()