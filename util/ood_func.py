import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import pinv, norm
from tqdm import tqdm

def energy_score(logits, T=1.0):
    """
        计算OOD能量得分的函数
        :param logits: 预测的logits值，大小为[batch_size, num_classes]
        :param T: 温度参数，默认为1.0
        :return: OOD能量得分
        """
    to_np = lambda x: x.data.cpu().numpy()
    prob = torch.from_numpy(logits)
    score = -to_np((T * torch.logsumexp(prob / T, dim=1)))
    return score


def odin_score(features, clip, w, b, temperature=100, noise_multiplier=0.001):
    """ODIN函数实现

    参数:
    logits (numpy.ndarray): 模型的输出logits
    features (numpy.ndarray): 输入样本的特征
    temperature (float): softmax温度超参数，默认为1000
    noise_multiplier (float): 扰动的噪声标准差，默认为0.001

    返回:
    (torch.Tensor): 对抗样本的置信度得分
    """
    # logits = torch.from_numpy(logits)
    # features = torch.from_numpy(features)
    # # 计算softmax后的输出
    # outputs = F.softmax(logits / temperature)
    # # 扰动原始输入
    # perturbed_features = features + noise_multiplier * torch.randn_like(features)
    # # 计算扰动后的logits
    # perturbed_logits = perturbed_features @ w.T + b
    # # 计算softmax后的扰动输出
    # perturbed_outputs = F.softmax(perturbed_logits * temperature)
    # # 计算对抗样本的置信度得分
    # confidence_score = (outputs - perturbed_outputs).abs().max(dim=-1)[0]
    #
    # return -confidence_score
    logits = np.clip(features, a_min=None, a_max=clip) @ w.T + b
    return -logsumexp(logits, axis=-1)

def msp_score(logits):
    softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    msp = np.max(softmax, axis=1)
    return -msp

# def residual_score(ref_model, x_test, y_test):
#     y_pred_test = ref_model.predict(x_test)
#     residual = np.abs(y_pred_test - y_test)
#     return np.mean(residual, axis=1)

def residual_score(logits, features, w, b):
    def gradnorm(x, w, b):
        fc = torch.nn.Linear(*w.shape[::-1])
        fc.weight.data[...] = torch.from_numpy(w)
        fc.bias.data[...] = torch.from_numpy(b)
        fc.cpu()

        x = torch.from_numpy(x).float().cpu()
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cpu()

        confs = []

        for i in tqdm(x):
            targets = torch.ones((1, 8)).cpu()
            fc.zero_grad()
            loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
            loss.backward()
            layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
            confs.append(layer_grad_norm)

        return np.array(confs)

    return gradnorm(features, w, b)
    # return -np.mean(np.abs(logits - np.mean(logits, axis=0)), axis=-1) + np.mean(np.abs(features - np.mean(features, axis=0)), axis=-1)

def get_mean_conv(feature_id_train, train_y, train_labels):
    print('computing classwise mean feature...')
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(len(train_labels))):  # 进度条库
        # fs = [feature_id_train[y_i] for y_i in range(train_y.shape[0]) if train_y[y_i] == i]
        # fs = feature_id_train[np.argwhere(train_y == i)]
        fs = feature_id_train[train_y == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    print('computing precision matrix...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    mean = torch.from_numpy(np.array(train_means)).cpu().float()
    prec = torch.from_numpy(ec.precision_).cpu().float()
    return mean, prec

def mahalanobis_score(feature, mean, prec):
    return np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
               torch.from_numpy(feature).cpu().float()])

def mahalanobis_score_(x, in_mean, in_cov):
    """
    计算输入数据的马氏距离得分，以评估其是否属于训练集中的数据。
    函数首先通过训练集中的数据计算出协方差矩阵和均值向量，并将其用于计算输入数据与训练集的马氏距离。
    马氏距离是一种考虑协方差矩阵的距离度量方法，用于度量输入数据与训练集数据之间的距离，即它们在同一分布下的距离
    :param x:
    :param in_mean: 均值
    :param in_cov: 协方差
    :return:
    """
    # 计算输入样本到训练集样本的Mahalanobis距离
    diff = x - in_mean
    inv_covmat = np.linalg.inv(in_cov)
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covmat).dot(diff[i])))
    return np.mean(md)

def maxlogit_socre(logits):
    # # 计算预测概率
    # probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    # # 选取最大的预测概率
    # max_probs = np.max(probs, axis=-1) * 100
    # # 返回 MaxLogit 得分
    # return -max_probs
    return -logits.max(axis=-1)

def nusa_score(logits, axis=1):
    """
    计算NuSA分数

    参数：
        logits: numpy.ndarray，模型对数据集的预测概率分布，shape为[num_samples, num_classes]
        axis: int，求取softmax的维度，默认为1

    返回值：
        nusa_score: float，NuSA分数
    """
    probs = softmax(logits, axis=axis)
    max_probs = np.max(probs, axis=axis) * 100
    # nusa_score = np.mean(max_probs)

    return -max_probs

def get_vl_param(feature_id_train, logit_id_train, DIM, w, b):
    # DIM = 100
    # 重新定义原点为  o := - (wT)+  * b ；Moore-Penrose
    u = -np.matmul(pinv(w), b)

    print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)  # 最大似然协方差估计器。
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')
    return alpha, NS, u

def vl_score(logit,feature, alpha, NS, u):

    # feature_id_val
    vlogit_id_val = norm(np.matmul(feature - u, NS), axis=-1) * alpha
    # energy_id_val = logsumexp(logit, axis=-1)
    energy_id_val = energy_score(logit)
    score_id = -vlogit_id_val - energy_id_val

    # logit_ = np.concatenate((logit, vlogit_id_val.reshape(-1, 1)), axis=1)
    # prob = softmax(logit_, axis=-1)
    # score_id = prob[:, -1]
    return -score_id

def residual_score_(feature, NS, u):
    return -norm(np.matmul(feature - u, NS), axis=-1)