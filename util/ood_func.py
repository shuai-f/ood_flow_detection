import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from scipy.special import logsumexp, softmax

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


def odin_score(logits, features, w, b, temperature=1000, noise_multiplier=0.001):
    """ODIN函数实现

    参数:
    logits (numpy.ndarray): 模型的输出logits
    features (numpy.ndarray): 输入样本的特征
    temperature (float): softmax温度超参数，默认为1000
    noise_multiplier (float): 扰动的噪声标准差，默认为0.001

    返回:
    (torch.Tensor): 对抗样本的置信度得分
    """
    logits = torch.from_numpy(logits)
    features = torch.from_numpy(features)
    # 计算softmax后的输出
    outputs = F.softmax(logits / temperature)
    # 扰动原始输入
    perturbed_features = features + noise_multiplier * torch.randn_like(features)
    # 计算扰动后的logits
    perturbed_logits = perturbed_features @ w.T + b
    # 计算softmax后的扰动输出
    perturbed_outputs = F.softmax(perturbed_logits / temperature)
    # 计算对抗样本的置信度得分
    confidence_score = (outputs - perturbed_outputs).abs().max(dim=-1)[0]

    return confidence_score

def msp_score(logits):
    softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    msp = 1 - np.max(softmax, axis=1)
    return msp

# def residual_score(ref_model, x_test, y_test):
#     y_pred_test = ref_model.predict(x_test)
#     residual = np.abs(y_pred_test - y_test)
#     return np.mean(residual, axis=1)

def residual_score(logits, features):
    return np.mean(np.abs(logits - np.mean(logits, axis=0)), axis=-1) + np.mean(np.abs(features - np.mean(features, axis=0)), axis=-1)


def mahalanobis_score(x, in_mean, in_cov):
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
    # 计算预测概率
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    # 选取最大的预测概率
    max_probs = np.max(probs, axis=-1)
    # 返回 MaxLogit 得分
    return -max_probs

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
    max_probs = np.max(probs, axis=axis)
    # nusa_score = np.mean(max_probs)

    return max_probs