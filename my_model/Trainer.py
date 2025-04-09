import numpy as np
import os
import matplotlib.pyplot as plt 
from my_model.evaluate import accuracy, loss
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

class Trainer:
    def __init__(self, model, optimizer, batch_size=128, epochs=50):
        """
        初始化 Trainer 类
        参数：
        - model: NeuralNetwork 类的实例
        - optimizer: 优化器实例 (如 SGD)
        - batch_size: 每个批次的样本数量，默认为 128
        - epochs: 训练的总轮次，默认为 50
        """
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

        # 历史记录
        self.train_accuracy_his = []
        self.val_accuracy_his = []
        self.train_loss_his = []
        self.val_loss_his = []

    def train(self, X_train, y_train, X_val, y_val):
        """
        训练神经网络模型
        参数：
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        """
        n = X_train.shape[0]
        best_accuracy = 0
        best_params = {}

        for epoch in range(self.epochs):
            # 更新学习率
            self.optimizer.update_learning_rate(epoch)

            # 打乱数据
            indices = np.arange(n)
            np.random.shuffle(indices)
            for i in range(0, n, self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, n)]
                X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                # 前向传播
                self.model.forward(X_batch)

                # 反向传播
                grads = self.model.backward(X_batch, y_batch)

                # 更新参数
                params = {'W1': self.model.W1, 'b1': self.model.b1,
                          'W2': self.model.W2, 'b2': self.model.b2,
                          'W3': self.model.W3, 'b3': self.model.b3}
                self.optimizer.step(params, grads)

            # 训练集、验证集上计算准确率
            train_accuracy = accuracy(self.model, X_train, y_train)
            val_accuracy = accuracy(self.model, X_val, y_val)
            self.train_accuracy_his.append(train_accuracy)
            self.val_accuracy_his.append(val_accuracy)
            print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}")

            # 训练集、验证集上计算损失
            train_loss = loss(self.model, X_train, y_train)
            val_loss = loss(self.model, X_val, y_val)
            self.train_loss_his.append(train_loss)
            self.val_loss_his.append(val_loss)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            # 更新最佳参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = {'W1': self.model.W1, 'b1': self.model.b1,
                               'W2': self.model.W2, 'b2': self.model.b2,
                               'W3': self.model.W3, 'b3': self.model.b3}
        
        print(f"Best Validation Accuracy: {best_accuracy:.4f}")
        # 最终获得的最优模型参数
        self.model.W1, self.model.b1 = best_params['W1'], best_params['b1']
        self.model.W2, self.model.b2 = best_params['W2'], best_params['b2']
        self.model.W3, self.model.b3 = best_params['W3'], best_params['b3']

    def plot_Accuracy(self):   
        os.makedirs('figs', exist_ok=True)  # 创建文件夹（若不存在）
        save_path = os.path.join('figs', 'acc.png')
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs+1), self.val_accuracy_his, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.show()

    def plot_loss(self):   
        os.makedirs('figs', exist_ok=True)  # 创建文件夹（若不存在）
        save_path = os.path.join('figs', 'loss.png')
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs+1), self.train_loss_his, label="Train Loss")
        plt.plot(range(1, self.epochs+1), self.val_loss_his, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.show()
    