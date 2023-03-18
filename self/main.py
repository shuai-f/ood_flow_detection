import argparse

import numpy as np

from contrast_learning.model import MLP, SimpleRNN, LSTM, GRU, CNN
import self_dataset
from ood_detection import LocalThreshold, VirtualLogit
from util.utils import list_to_str

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
        from main.self_dataset import train_labels, ood_labels
        trainLabels = train_labels
        oodLabels = ood_labels
        train_x, train_y, test_x, test_y, ood_x, ood_y = self_dataset.read_data(oodLabels)
        num_pixel = 9 * 9
        train_x = train_x.reshape(-1, num_pixel).astype(np.float32)
        test_x = test_x.reshape(-1, num_pixel).astype(np.float32)
    else :
        print("Unknown Dataset Name :{}".format(args.data))
        return

    print(train_x.shape, test_x.shape)
    model_name = '{}_{}-bs_{}-lr_{}'.format(
        args.data, args.model, args.batch_size, args.lr)

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

    # model = MLP(normalize=True, activation=args.activation)
    model.build(input_shape=(None, num_pixel, ))
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

    # ood_x = get_ood_dict(ood_x, ood_y.numpy(), oodLabels)
    # VirtualLogit(ood_x, ood_y, train_x, train_y, test_x, test_y, mlp, w, b, trainLabels, oodLabels)