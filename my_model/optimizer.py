import numpy as np

class SGD:
    def __init__(self, lr=0.1, decay_rate=0.01):
        """
        初始化 SGD 优化器
        参数：
        - lr: 初始学习率
        - decay_rate: 学习率衰减率
        """
        self.initial_lr = lr
        self.lr = lr
        self.decay_rate = decay_rate

    def update_learning_rate(self, epoch):
        self.lr = self.initial_lr * np.exp(-self.decay_rate * epoch)

    def step(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 
