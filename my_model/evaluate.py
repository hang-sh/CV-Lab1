import numpy as np
from my_model.NeuralNetwork import NeuralNetwork

def cross_entropy_loss(y_pred, y_true):
    """
    计算平均交叉熵损失
    参数：
    - y_pred: shape (m, C)
        每一行表示一个样本对 C 个类别的预测概率
    - y_true: shape (m, C)
        每一行是该样本的真实标签(one-hot 编码)
    """
    m = y_true.shape[0]  
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

# 总损失函数
def loss(model:NeuralNetwork, X, y_true):
    y_pred = model.forward(X)
    # 交叉熵损失
    loss = cross_entropy_loss(y_pred, y_true)
    # L2 正则化
    loss += 0.5 * model.reg_lambda * (np.sum(model.W1**2) + np.sum(model.W2**2) + np.sum(model.W3**2))  
    return loss

def accuracy(model:NeuralNetwork, X, y_true):
    y_pred = model.forward(X)
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))