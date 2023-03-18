import moore
from moore import train_x, train_y, test_x, test_y, ood_x, ood_y, train_labels, ood_labels, num_classes, num_pixels
from util import ml, plot
from util.ml import simple_CNN, load_CNN, Bayes, contrast_learning_CNN, DecisionTr, LSTM, load_DecisionTr, \
    local_threshold

if __name__ == '__main__':
    # moore.init(1)
    ml.config(numClass=num_classes, numPixels=num_pixels, ood_X=ood_x, ood_Y=ood_y, train_Labels=train_labels, ood_Labels=ood_labels)
    # baseline(train_x, train_y, test_x, test_y)
    # simple_CNN(train_x, train_y, test_x, test_y)
    local_threshold(train_x, train_y, test_x, test_y)
    # contrast_learning_CNN()

    # DecisionTr(train_x, train_y, test_x, test_y)
    # load_DecisionTr()

    # plt_image(train_x, train_y)
    print("-----\n")
    # RandomForest(train_x, train_y, test_x, test_y)
    # LSTM(train_x, train_y, test_x, test_y)