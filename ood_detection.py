import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from numpy.linalg import pinv, norm
from sklearn.covariance import EmpiricalCovariance
from sklearn import metrics
from scipy.special import logsumexp, softmax
import pandas as pd

import util.utils


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
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def Msp(val_x, val_y, ood_x, ood_y, model, train_labels, ood_labels):
    """
    Baseline 基线方法，通过设定softmax阈值确定样本是否为ood样本，
    :param val_x:
    :param val_y:
    :param ood_x:
    :param ood_y:
    :param model:
    :param train_labels:
    :param ood_labels:
    :return:
    """
    print("Baseline MSP : ==================================")
    # 计算MSP阈值
    # score(Xtest) = Pmax = max{P1,P2, ... Pc}
    # print("\ncalculate softmax threshold : ----------------")
    # softmax_max = np.max(predict_y, axis=1)
    # print(softmax_max, softmax_max.shape)
    # # TODO 阈值取测试集最大还是最小
    # softmax_threshold = np.max(softmax_max, axis=0)
    # print("threshold : " + str(softmax_threshold))
    # 阈值检测
    print("\ndetection : -------------------")
    # threshold = softmax_threshold
    ood_X = tf.reshape(ood_x, [-1, 16, 16, 1])
    ood_predict_y = model.predict(ood_X)  # 预测概率
    ood_pred_y = np.argmax(ood_predict_y, axis=1)  # 预测类别索引
    softmax_max = np.max(ood_predict_y, axis=1)  # 预测类别对应置信度

    val_X = tf.reshape(val_x, [-1, 16, 16, 1])
    val_predict_y = model.predict(val_X)
    val_softmax_max = np.max(val_predict_y, axis=1)

    ood_error = []
    ood_thresholds = [0.1, 0.01, 0.001, 0.0001]
    for i in ood_thresholds:
        # threshold += 0.1
        threshold = 1.0 - i
        print("threshold : " + str(threshold))
        ood_list = np.where(softmax_max > threshold)[0]
        print("ood_list : ")
        print(ood_list)
        print("softmax prob : ")
        print([softmax_max[j] for j in ood_list])
        print([ood_labels[ood_y[j]] for j in ood_list])
        print([train_labels[ood_pred_y[j]] for j in ood_list])
        print("ind -> ood : %.2f%%" % (len(np.where(val_softmax_max < threshold)[0]) / len(val_y) * 100))
        print("ood -> ind : %.2f%%" % (len(ood_list) / len(ood_y) * 100))
        print('------------------------------')
        ood_error.append((threshold, len(ood_list) / len(ood_y)))
    print("MSP result :")
    print(ood_error)


def VirtualLogit(ood_x: Tensor, ood_y: Tensor, train_x: Tensor, train_y: Tensor, val_x: Tensor, val_y: Tensor,
                 model: tf.keras.Sequential, train_labels: list, ood_labels: list):
    """
    虚拟对数方法
    C 分类问题， 训练集大小K， 通过特征向量针对主子空间的投影，
    计算原始训练样本的logits， 
    l_0 = alpha * || exp(x, P⊥) ||
    alpha = ∑i=1,..k max j=1,..,C [l(i, j)] / ∑i=1,..k || exp(x, P⊥) ||
    :return:
    """

    # 获取模型权重和偏置
    print("\nweights and bias : -------------------")
    weight_Dense_1, bias_Dense_1 = model.get_layer(index=-2).get_weights()
    # print(weight_Dense_1.shape, bias_Dense_1.shape)
    weight_Dense_2, bias_Dense_2 = model.get_layer(index=-1).get_weights()
    # print(weight_Dense_2.shape, bias_Dense_2.shape)
    # W ∈ RN×C
    w = weight_Dense_1 @ weight_Dense_2
    w = w.T
    # b ∈ RC
    b = bias_Dense_1 @ weight_Dense_2 + bias_Dense_2
    print(w, w.shape)
    print(b, b.shape)
    ood_X = tf.reshape(ood_x, [-1, 16, 16, 1])
    val_X = tf.reshape(val_x, [-1, 16, 16, 1])  # X = [x1, x2, ... , xk]
    prob_y = model.predict(val_X)

    # 预处理
    # feature
    feature_id_train = train_x.numpy()
    feature_id_val = val_x.numpy()
    feature_ood = ood_x.numpy()
    # logit
    print('computing logits')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_ood = feature_ood @ w.T + b
    #
    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_ood = softmax(logit_ood, axis=-1)

    df = pd.DataFrame(columns=['method', 'oodset', 'auroc', 'fpr'])
    dfs = []
    recall = 0.95  # 召回率
    name = util.utils.list_to_str(ood_labels)  # ood class

    print("MSP : ==============================")
    method = 'MSP'
    result = []
    score_id = softmax_id_val.max(axis=-1)
    score_ood = softmax_ood.max(axis=-1)
    auc_ood = auc(score_id, score_ood)[0]
    fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    print("MaxLogit : ============================")
    method = 'MaxLogit'
    result = []
    score_id = logit_id_val.max(axis=-1)
    score_ood = logit_ood.max(axis=-1)
    auc_ood = auc(score_id, score_ood)[0]
    fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    print("Energy : ============================")
    method = 'Energy'
    result = []
    score_id = logsumexp(logit_id_val, axis=-1)
    score_ood = logsumexp(logit_ood, axis=-1)
    auc_ood = auc(score_id, score_ood)[0]
    fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    print("Energy+React : ============================")
    method = 'Energy+React'
    result = []

    clip_quantile = 0.99
    clip = np.quantile(feature_id_train, clip_quantile)
    print(f'clip quantile {clip_quantile}, clip {clip:.4f}')

    logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_val_clip, axis=-1)
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    auc_ood = auc(score_id, score_ood)[0]
    fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
    result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
    print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    print("Residual : ============================")
    method = 'Residual'
    # DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    DIM = 0
    while DIM < 200:
        result = []
        # 重新定义原点为  o := - (wT)+  * b ；Moore-Penrose
        u = -np.matmul(pinv(w), b)
        print("choose DIM : " + str(DIM))

        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)

        score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(dict(method=method + "-DIM:" + str(DIM), oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        DIM += 40

    print("Virtual Logit : ==================================")
    method = 'Virtual Logit'
    # 256维特征
    DIM = 0
    while DIM < 200:
        result = []
        # 重新定义原点为  o := - (wT)+  * b ；Moore-Penrose
        u = -np.matmul(pinv(w), b)
        print("choose DIM : " + str(DIM))

        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)  # 最大似然协方差估计器。
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        print('computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')

        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        score_id = -vlogit_id_val + energy_id_val

        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(dict(method=method + '-DIM:' + str(DIM), oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        #
        # for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(), feature_oods.values()):
        #     energy_ood = logsumexp(logit_ood, axis=-1)
        #     vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        #     score_ood = -vlogit_ood + energy_ood
        #     auc_ood = auc(score_id, score_ood)[0]
        #     fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
        DIM += 40

    # df.to_csv("./output/splits/splits_2/preference.csv")


    file = "./output/preference/preference_%s.csv" % name
    with open(file, "a") as f :
        f.write("num,method,oodset,auroc,fpr95\n")
        for i in range(0, len(dfs)):
            df = dfs[i].iloc[0]  # 获取第一行数据
            f.write("%d,%s,%s,%.2f%%,%.2f%%\n" % (i + 1, df.method, df.oodset, df.auroc * 100, df.fpr * 100))


def SelfContrast():
    """
    自适应对比学习算法
    :return:
    """


