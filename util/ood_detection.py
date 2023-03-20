import copy
import math
import os
import time

import numpy as np
import tensorflow as tf
import torch
from scipy import integrate
from sklearn.metrics import confusion_matrix
from tensorflow import Tensor
from numpy.linalg import pinv, norm
from sklearn.covariance import EmpiricalCovariance
from sklearn import metrics
from scipy.special import logsumexp, softmax
import pandas as pd
from tqdm import tqdm

import util.utils
from util.metrics import fpr_recall, auc
from util.ood_func import energy_score, msp_score, maxlogit_socre, nusa_score, odin_score, residual_score
from util.utils import list_to_str, plt_line, plt_alpha

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

def get_ood_dict(ood_x_numpy,ood_y_numpy, ood_labels):
    '''
    get ood dict : {num_label, ood_numpy}
    :param ood_x:
    :param ood_y:
    :return:
    '''
    ood_X = {}
    print('ood_x_numpy shape:')
    print(ood_x_numpy.shape)
    print(ood_x_numpy[0].T.shape)
    for i in range(ood_y_numpy.shape[0]):  # 索引
        if ood_X.get(ood_y_numpy[i]) is None:
            ood_X[ood_y_numpy[i]] = [ood_x_numpy[i]]
        else:
            ood_X[ood_y_numpy[i]] = np.append(ood_X[ood_y_numpy[i]], [ood_x_numpy[i]], axis=0)
    for key, value in ood_X.items():
        print(key, value.shape)
    return {ood_labels[name]:feature_ood for name, feature_ood in ood_X.items()}

def VirtualLogit(feature_oods, ood_y, feature_id_train, train_y, feature_id_val, val_y,
                 model, w, b, train_labels: list, ood_labels: list):
    """
    虚拟对数方法
    C 分类问题， 训练集大小K， 通过特征向量针对主子空间的投影，
    计算原始训练样本的logits， 计算样本的vlogit， 共同计算softmax
    l_0 = alpha * || exp(x, P⊥) ||
    alpha = ∑i=1,..k max j=1,..,C [l(i, j)] / ∑i=1,..k || exp(x, P⊥) ||
    :param feature_oods: features before softmax layer , type numpy
    :param ood_y: [num_label]
    :param feature_id_train: features before softmax layer , type numpy
    :param train_y: [num_label]
    :param feature_id_val: features before softmax layer , type numpy
    :param val_y: [num_label]
    :param model:
    :param w: softmax layer weight
    :param b: softmax layer bias
    :param train_labels: [str_label]
    :param ood_labels: [str_label]
    :return:
    """
    # feature
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
    while DIM <= feature_id_train.shape[1] * 5 // 6:
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
    fpr_result, auroc_result, alpha_list = {},{},{'alpha':[], 'mean_auroc':[], 'mean_fpr':[]}
    DIM_set = []
    while DIM <= feature_id_train.shape[1] * 5 // 6:
        result = []
        # 重新定义原点为  o := - (wT)+  * b ；Moore-Penrose
        u = -np.matmul(pinv(w), b)
        print("choose DIM : " + str(DIM))
        DIM_set.append(DIM)

        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)  # 最大似然协方差估计器。
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        DIM += 5

        print('computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')
        alpha_list['alpha'].append(alpha)
        if math.isinf(alpha):
            continue

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
            if auroc_result.get(name) is None:
                auroc_result[name] = [auc_ood]
            else:
                auroc_result[name].append(auc_ood)
            if fpr_result.get(name) is None:
                fpr_result[name] = [fpr_ood]
            else:
                fpr_result[name].append(fpr_ood)
        df = pd.DataFrame(result)
        dfs.append(df)
        print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
        alpha_list['mean_auroc'].append(df.auroc.mean())
        alpha_list['mean_fpr'].append(df.fpr.mean())
    # plt_line('Performance of different DIM', 'DIM', 'AUROC', DIM_set, ood_result)
    # 画折线图
    plt_line('Auroc of different DIM', 'DIM', 'AUROC', DIM_set, auroc_result)

    plt_line('Fpr of different DIM', 'DIM', 'FPR', DIM_set, fpr_result)

    plt_alpha(DIM_set, alpha_list['alpha'], alpha_list['mean_auroc'], alpha_list['mean_fpr'],'Mean performance of different alpha')


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
        fs = feature_id_train[train_y == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    print('computing precision matrix...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

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

    write_df(dfs, ood_labels)

def write_df(dfs, ood_labels):
    time_now = time.strftime("%Y-%m-%d", time.localtime())
    path = "./output/preference/" + '/' + time_now
    if not os.path.exists(path):
        os.makedirs(path)
        print('文件夹创建完成  ' + path)
    path += '/' + time.strftime("%H-%M-%S", time.localtime()) + '_'
    file = path + "preference_%s.csv" % list_to_str(ood_labels)
    with open(file, "a") as f:
        _str = ""
        _preference = ""
        for i in ood_labels:
            _str += i + ',,'
            _preference += "auroc,fpr,"
        f.write(",oodset," + _str + "Average,,\n")
        f.write("num,method," + _preference + 'auroc,fpr\n')
        # for column in range(len(ood_labels)):
        #     for i in range(0, len(dfs)):
        #         df = dfs[i].iloc[column]  # 获取第一行数据
        #         f.write("%d,%s,%s,%.2f%%,%.2f%%\n" % (i + 1, df.method, df.oodset, df.auroc * 100, df.fpr * 100))

        len_ood = len(ood_labels)
        for i in range(0, len(dfs)):
            df = dfs[i]
            template = str(i) + ',' + df.iloc[0].method + ','
            auroc, fpr = 0.0,0.0
            for ood_dataset in ood_labels:
                for column in range(len_ood):
                    df_line = df.iloc[column]
                    if ood_dataset == df_line.oodset:
                        auroc += df_line.auroc
                        fpr += df_line.fpr
                        template += "%.2f%%,%.2f%%," %(df_line.auroc * 100, df_line.fpr * 100)
                        continue
            template += "%.2f%%,%.2f%%,\n" %(auroc * 100 / len_ood, fpr * 100 / len_ood)
            print(template)
            f.write(template)

def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray,
                            plus=1e-10) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])

    threshold = 0.
    tp, fp, fn, tn= len(unseen_m_dist), len(seen_m_dist), 0, 0

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)

    def compute_auroc(tp, fp, fn, tn):
        fpr = fp / (fp + tn + 1e-10)
        tpr = tp / (tp + fn + 1e-10)
        auc_roc, _ = integrate.simps(y=tpr, x=fpr)
        return auc_roc

    f1 = compute_f1(tp, fp, fn)
    # auroc = compute_auroc(tp, fp, fn, tn)

    for m_dist, label in lst:   #
        if label == "seen":  # fp -> tn
            fp -= 1
            tn += 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + plus

    # print("estimated threshold:", threshold)
    print("local f1 {}, seen {}, unseen {}, threshold {}".format(f1, len(seen_m_dist), len(unseen_m_dist), threshold))
    return threshold

def get_score(cm):
    fs = []
    ps = []
    rs = []
    n_class = cm.shape[0]
    correct = []
    total = []
    for idx in range(n_class):
        TP = cm[idx][idx]
        correct.append(TP)
        total.append(cm[idx].sum())
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)
        ps.append(p * 100)
        rs.append(r * 100)

    f = np.mean(fs).round(2)
    p = np.mean(ps).round(2)
    r = np.mean(rs).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    p_unseen = round(ps[-1], 2)
    r_unseen = round(rs[-1], 2)
    f_unseen = round(fs[-1], 2)
    print(f"Overall(macro): , f:{f},  p:{p}, r:{r}")
    print(f"Seen(macro): , f:{f_seen}, p:{p_seen}, r:{r_seen}")
    print(f"=====> Unseen <=====: , f:{f_unseen}, p:{p_unseen}, r:{r_unseen}\n")

    return f, f_seen,  p_seen, r_seen, f_unseen,  p_unseen, r_unseen


def LocalThreshold(feature_oods, ood_y, feature_id_train, train_y, feature_id_val, val_y,
                 model, w, b, train_labels: list, ood_labels: list, ood_func=energy_score, T=1.0):
    """
    局部阈值
    :return:
    """
    # feature
    # logit
    print('computing logits...')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    # logit_oods = {name: feature_ood @ w.T + b for name, feature_ood in feature_oods.items()}
    logit_oods = feature_oods @ w.T + b
    #
    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    # softmax_oods = {name: softmax(logit_ood, axis=-1) for name, logit_ood in logit_oods.items()}
    softmax_oods = softmax(logit_oods, axis=1)

    # tpr = 0.95
    # 给予能量
    print("{} : ============================".format(ood_func))
    # 计算OOD分数
    if ood_func == energy_score:
        score_id_train = ood_func(logit_id_train, T)
        score_id_val = ood_func(logit_id_val, T)
        score_ood = ood_func(logit_oods, T)
    elif ood_func == msp_score or ood_func == maxlogit_socre or ood_func == nusa_score:
        score_id_train = ood_func(logit_id_train)
        score_id_val = ood_func(logit_id_val)
        score_ood = ood_func(logit_oods)
    elif ood_func == residual_score:
        score_id_train = ood_func(logit_id_train, feature_id_train)
        score_id_val = ood_func(logit_id_val, feature_id_val)
        score_ood = ood_func(logit_oods, feature_oods)
    elif ood_func == odin_score:
        score_id_train = ood_func(logit_id_train, feature_id_train, w, b)
        score_id_val = ood_func(logit_id_val, feature_id_val, w, b)
        score_ood = ood_func(logit_oods, feature_oods, w, b)
    else:
        print("Unknown ood func")
        return

    print('score_id_train : {}'.format(score_id_train))
    print('score_ood : {}'.format(score_ood))

    # y_pred
    y_pred_id_train = np.argmax(softmax_id_train, axis=-1)
    y_pred_id_val = np.argmax(softmax_id_val, axis=-1)
    y_pred_ood = np.argmax(softmax_oods, axis=-1)

    # val and ood set y_true
    y_true_all = np.append(val_y, np.array([len(train_labels)] * len(feature_oods)))
    # 原始模型输出矩阵
    print("origin model predict : ")
    y_pred_all = np.append(y_pred_id_val, y_pred_ood)
    # print(y_true_all.shape)
    # y_predict = model.predict(np.append(feature_id_val, feature_oods).reshape((-1,16*16)))
    # y_pred = np.argmax(y_predict, axis=-1)
    # print(y_pred.shape)
    # print(confusion_matrix(y_true_all, y_pred))
    # 原矩阵
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(cm)

    plus_ = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for plus in plus_:
        # 训练集和分布外数据评估一个最佳全局阈值
        ori_better_threshold = estimate_best_threshold(score_id_train, score_ood, plus)
        print("origin global threshold : {}".format(ori_better_threshold))
        print("model predict under global threshold :")
        y_val = copy.deepcopy(y_pred_id_val)
        y_ood = copy.deepcopy(y_pred_ood)
        y_pred_val_score_threshold = score_id_val - ori_better_threshold
        y_val[np.where(y_pred_val_score_threshold > 0)] = len(train_labels)
        y_pred_ood_score_threshold = score_ood - ori_better_threshold
        y_ood[np.where(y_pred_ood_score_threshold > 0)] = len(train_labels)
        y_pred_all = np.append(y_val, y_ood)
        cm = confusion_matrix(y_true_all, y_pred_all)
        print(cm)
        print("f, f_seen, p_seen, r_seen, f_unseen, p_unseen, r_unseen")
        get_score(cm)

    print("model predict under local threshold :")
    y_pred_val_thresholds = copy.deepcopy(y_pred_id_val)
    y_pred_ood_thresholds = copy.deepcopy(y_pred_ood)
    thresholds = {}
    # 计算局部阈值
    for label in range(len(train_labels)):
        ypred_val_indexs = np.argwhere(y_pred_id_train == label)
        ypred_val_score = score_id_train[ypred_val_indexs]
        ypred_ood_indexs = np.argwhere(y_pred_ood == label)
        # print(ypred_ood_indexs, len(ypred_ood_indexs))
        ypred_ood_score = score_ood[ypred_ood_indexs]
        threshold = estimate_best_threshold(ypred_val_score, ypred_ood_score)
        if threshold == 0:
            threshold = ori_better_threshold
        # 给出对应每一个分布内的类局部阈值
        thresholds[label] = [threshold]
        pred_indexs = np.argwhere(y_pred_id_val == label)
        y_pred_val_thresholds[pred_indexs] = threshold
        pred_ood_index = np.argwhere(y_pred_ood == label)
        y_pred_ood_thresholds[pred_ood_index] = threshold
    print("thresholds : {}".format(thresholds))

    y_pred_val_score_threshold = score_id_val - y_pred_val_thresholds
    y_pred_id_val[np.where(y_pred_val_score_threshold > 0)] = len(train_labels)
    y_pred_ood_score_threshold = score_ood - y_pred_ood_thresholds
    y_pred_ood[np.where(y_pred_ood_score_threshold > 0)] = len(train_labels)
    # 取验证集和ood_set
    y_pred_all = np.append(y_pred_id_val, y_pred_ood)
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(cm)
    print("f, f_seen, p_seen, r_seen, f_unseen, p_unseen, r_unseen")
    print(get_score(cm))

