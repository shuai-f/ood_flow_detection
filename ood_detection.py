import numpy as np
import tensorflow as tf
import torch
from tensorflow import Tensor
from numpy.linalg import pinv, norm
from sklearn.covariance import EmpiricalCovariance
from sklearn import metrics
from scipy.special import logsumexp, softmax
import pandas as pd
from tqdm import tqdm

import util.utils
from util.utils import list_to_str

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
    print('fpr , tpr')
    print(fpr.shape, tpr.shape)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    print('precision , recall')
    print(precision_in, recall_in)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)
    print(precision_out, recall_out)

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
                 model: tf.keras.Sequential, w, b, train_labels: list, ood_labels: list):
    """
    虚拟对数方法
    C 分类问题， 训练集大小K， 通过特征向量针对主子空间的投影，
    计算原始训练样本的logits， 计算样本的vlogit， 共同计算softmax
    l_0 = alpha * || exp(x, P⊥) ||
    alpha = ∑i=1,..k max j=1,..,C [l(i, j)] / ∑i=1,..k || exp(x, P⊥) ||
    :return:
    """

    # 预处理
    ood_X = {}
    ood_y_numpy = ood_y.numpy()
    ood_x_numpy = ood_x.numpy()
    print('ood_x_numpy shape:')
    print(ood_x_numpy.shape)
    print(ood_x_numpy[0].T.shape)
    for i in range(ood_y_numpy.shape[0]): # 索引
        if ood_X.get(ood_y_numpy[i]) is None:
            ood_X[ood_y_numpy[i]] = [ood_x_numpy[i]]
        else:
            ood_X[ood_y_numpy[i]] = np.append(ood_X[ood_y_numpy[i]], [ood_x_numpy[i]], axis=0)
    for key, value in ood_X.items():
        print(key, value, value.shape)
    # feature
    feature_id_train = train_x.numpy()
    feature_id_val = val_x.numpy()
    feature_oods = {ood_labels[name]:feature_ood for name, feature_ood in ood_X.items()}
    # logit
    print('computing logits')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_oods = {name:feature_ood @ w.T + b for name, feature_ood in feature_oods.items()}
    #
    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_oods = {name : softmax(logit_ood, axis=-1) for name, logit_ood in logit_oods.items()}

    df = pd.DataFrame(columns=['method', 'oodset', 'auroc', 'fpr'])
    dfs = []
    tpr = 0.95  # 召回率
    name = util.utils.list_to_str(ood_labels)  # ood class

    print("MSP : ==============================")
    method = 'MSP'
    result = []
    score_id = softmax_id_val.max(axis=-1)
    for name, softmax_ood in softmax_oods.items():
        score_ood = softmax_ood.max(axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    print("MaxLogit : ============================")
    method = 'MaxLogit'
    result = []
    score_id = logit_id_val.max(axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logit_ood.max(axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    print("Energy : ============================")
    method = 'Energy'
    result = []
    score_id = logsumexp(logit_id_val, axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logsumexp(logit_ood, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
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
    for name, feature_ood in feature_oods.items():
        logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
        score_ood = logsumexp(logit_ood_clip, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
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

        for name, feature_ood in feature_oods.items():
            # logit_ood = logit_oods[name]
            score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
            result.append(dict(method=method + "-DIM:" + str(DIM), oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        DIM += 20

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

        # feature_id_val
        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        # print(energy_id_val, energy_id_val.shape)
        score_id = -vlogit_id_val + energy_id_val
        # print(score_id, score_id.shape)

        # for name, logit_ood, feature_ood in zip(ood_labels, logit_oods.values(), feature_oods.values()):
        for name, feature_ood in feature_oods.items():
            logit_ood = logit_oods[name] # shape = (n, num_classes)
            # print(logit_ood, logit_ood.shape)
            energy_ood = logsumexp(logit_ood, axis=-1)
            vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
            # print(vlogit_ood,energy_ood) # shape = (n, )
            score_ood = -vlogit_ood + energy_ood
            auc_ood = auc(score_id, score_ood)[0]
            fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
            result.append(dict(method=method + '-DIM:' + str(DIM), oodset=name, auroc=auc_ood, fpr=fpr_ood))
            print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        #
        # for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(), feature_oods.values()):
        #     energy_ood = logsumexp(logit_ood, axis=-1)
        #     vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        #     score_ood = -vlogit_ood + energy_ood
        #     auc_ood = auc(score_id, score_ood)[0]
        #     fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
        #     result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        #     print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        df = pd.DataFrame(result)
        dfs.append(df)
        # print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
        DIM += 20

    # df.to_csv("./output/splits/splits_2/preference.csv")
    print("Mahalanobis : ==================================")
    method = 'Mahalanobis'
    result = []

    # train_labels = np.array([int(line.rsplit(' ', 1)[-1]) for line in train_labels], dtype=int)
    print('computing classwise mean feature...')
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(8)): # 进度条库
        # fs = [feature_id_train[y_i] for y_i in range(train_y.shape[0]) if train_y[y_i] == i]
        fs = feature_id_train[train_y.numpy() == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    print('computing precision matrix...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    print('go to gpu...')
    mean = torch.from_numpy(np.array(train_means)).cpu().float()
    prec = torch.from_numpy(ec.precision_).cpu().float()

    score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                          tqdm(torch.from_numpy(feature_id_val).cpu().float())])
    for name, feature_ood in feature_oods.items():
        score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                               tqdm(torch.from_numpy(feature_ood).cpu().float())])
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, tpr)
        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    file = "./output/preference/preference_%s.csv" % list_to_str(ood_labels)
    with open(file, "a") as f:
        f.write("num,method,oodset,auroc,fpr\n")
        for column in range(len(ood_labels)):
            for i in range(0, len(dfs)):
                df = dfs[i].iloc[column]  # 获取第一行数据
                f.write("%d,%s,%s,%.2f%%,%.2f%%\n" % (i + 1, df.method, df.oodset, df.auroc * 100, df.fpr * 100))

def SelfContrast():
    """
    自适应对比学习算法
    :return:
    """
    pass
