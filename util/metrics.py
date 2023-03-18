import numpy as np
from sklearn import metrics
import tensorflow as tf

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    # print('conf')
    print(conf, conf.shape)
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    # print('fpr {}, tpr {}'.format(fpr.shape, tpr.shape))
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    # print('precision {}, {}\n, recall {}, {}'.format(precision_in, precision_in.shape, recall_in, recall_in.shape))
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)
    # print(precision_out, recall_out)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)
    print("auroc {}, aupr_in {}, aupr_out {}".format(auroc, aupr_in, aupr_out))
    return auroc, aupr_in, aupr_out


def f1(y_hat, y_true, model='multi'):
    '''
    输入张量y_hat是输出层经过sigmoid激活的张量
    y_true是label{0,1}的集和
    model指的是如果是多任务分类，single会返回每个分类的f1分数，multi会返回所有类的平均f1分数（Marco-F1）
    如果只是单个二分类任务，则可以忽略model
    '''
    epsilon = 1e-7
    y_hat = tf.round(y_hat)  # 将经过sigmoid激活的张量四舍五入变为0，1输出

    tp = tf.reduce_sum(tf.cast(y_hat * y_true, 'float'), axis=0)
    # tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast(y_hat * (1 - y_true), 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast((1 - y_hat) * y_true, 'float'), axis=0)

    p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    if model == 'single':
        return f1
    if model == 'multi':
        return tf.reduce_mean(f1)

def f1_numpy(y_pred, y_true, THRESHOLD=0.5):
    '''
    y_hat是未经过sigmoid函数激活的
    输出的f1为Marco-F1
    '''

    epsilon = 1e-7
    y_pred = y_pred > THRESHOLD
    y_pred = np.int8(y_pred)
    tp = np.sum(y_pred == y_true)
    fp = np.sum(y_pred * (1 - y_true), axis=0)
    fn = np.sum((1 - y_pred) * y_true, axis=0)

    p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)

    return np.mean(f1)