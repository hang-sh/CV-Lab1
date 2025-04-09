import numpy as np
import os
import pickle

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    return np.exp(x-x_max)/ np.sum(np.exp(x-x_max), axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, input, hidden_1, hidden_2, output, activation = 'relu', reg_lambda=0.001):
        """
        初始化神经网络结构和训练参数
        参数说明：
        - input: 输入层大小
        - hidden_1: 第一隐藏层大小
        - hidden_2: 第二隐藏层大小
        - output: 输出层大小
        - activation: 使用的激活函数类型，默认为 ReLU
        - reg_lambda: L2 正则化系数，默认值为 0.001
        """
        # 网络结构参数
        self.input = input
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.output = output 
        
        # 训练超参数
        self.activation = activation
        self.reg_lambda = reg_lambda

        # 训练参数初始化(He 初始化)
        self.W1 = np.random.randn(self.input, self.hidden_1) * np.sqrt(2. / self.input)
        self.b1 = np.zeros((1, self.hidden_1))

        self.W2 = np.random.randn(self.hidden_1, self.hidden_2) * np.sqrt(2. / self.hidden_1)
        self.b2 = np.zeros((1, self.hidden_2))

        self.W3 = np.random.randn(self.hidden_2, self.output) * np.sqrt(2. / self.hidden_2)
        self.b3 = np.zeros((1, self.output))

    def activate(self, y, activation):
        if activation == 'relu':
            return relu(y)
        elif activation == 'sigmoid':
            return sigmoid(y)
        elif activation == 'leaky_relu':
            return leaky_relu(y)
        else:
            raise ValueError("Unsupported activation function")
    
    def activate_derivative(self, y, activation):
        if activation == 'relu':
            return relu_derivative(y)
        elif activation == 'sigmoid':
            return sigmoid_derivative(y)
        elif activation == 'leaky_relu':
            return leaky_relu_derivative(y)
        else:
            raise ValueError("Unsupported activation function")

    # 前向计算
    def forward(self, X):
        self.y1 = np.dot(X, self.W1) + self.b1
        self.z1 = self.activate(self.y1, self.activation)
        self.y2 = np.dot(self.z1, self.W2) + self.b2
        self.z2 = self.activate(self.y2, self.activation)
        self.y3 = np.dot(self.z2, self.W3) + self.b3
        self.z3 = softmax(self.y3)

        return self.z3
    
    # 反向传播计算梯度
    def backward(self, X, y_true):
        m = y_true.shape[0]
        grads = {}

        dY3 = self.z3 - y_true
        grads['W3'] = np.dot(self.z2.T, dY3) / m + self.reg_lambda * self.W3
        grads['b3'] = np.sum(dY3, axis=0, keepdims=True) / m

        dZ2 = np.dot(dY3, self.W3.T)
        dY2 = dZ2 * self.activate_derivative(self.y2, self.activation)
        grads['W2'] = np.dot(self.z1.T, dY2) / m + self.reg_lambda * self.W2
        grads['b2'] = np.sum(dY2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dY2, self.W2.T)
        dY1 = dZ1 * self.activate_derivative(self.y1, self.activation)
        grads['W1'] = np.dot(X.T, dY1) / m + self.reg_lambda * self.W1
        grads['b1'] = np.sum(dY1, axis=0, keepdims=True) / m

        return grads

    # 保存模型
    def save_model(self, filename="model.pkl", directory="saved_models"):
        os.makedirs(directory, exist_ok=True)  # 创建文件夹（若不存在）
        filepath = os.path.join(directory, filename)

        model_data = {
        'input': self.input,
        'hidden1': self.hidden_1,
        'hidden2': self.hidden_2,
        'output': self.output,
        'activation': self.activation,
        'reg_lambda': self.reg_lambda,
        'W1': self.W1,
        'b1': self.b1,
        'W2': self.W2,
        'b2': self.b2,
        'W3': self.W3,
        'b3': self.b3   }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    # 加载模型
    @classmethod
    def load_model(self, filepath="saved_models/model.pkl"):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    
        # 用加载的超参数创建一个新模型
        model = NeuralNetwork(
            input=model_data['input'],
            hidden_1=model_data['hidden1'],
            hidden_2=model_data['hidden2'],
            output=model_data['output'],
            activation=model_data['activation'],
            reg_lambda=model_data['reg_lambda'],
        )
    
        # 恢复权重和偏置
        model.W1 = model_data['W1']
        model.b1 = model_data['b1']
        model.W2 = model_data['W2']
        model.b2 = model_data['b2']
        model.W3 = model_data['W3']
        model.b3 = model_data['b3']

        return model
