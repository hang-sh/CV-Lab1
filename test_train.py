from my_model.NeuralNetwork import NeuralNetwork
from load_data import load_data
from my_model.optimizer import SGD
from my_model.Trainer import Trainer

# 导入数据
X_train, y_train, X_val, y_val, _, _ = load_data()

input = X_train.shape[1]
output = y_train.shape[1]

# 初始化模型
model = NeuralNetwork(input=input, hidden_1=512, hidden_2=64, output=output, activation='leaky_relu', reg_lambda=0.001)
# 初始化优化器
optimizer = SGD(lr=0.03, decay_rate=0.01)
# 初始化 Trainer
trainer = Trainer(model=model, optimizer=optimizer, batch_size=128, epochs=30)
# 开始训练
trainer.train(X_train, y_train, X_val, y_val)
model.save_model()

# 绘制 loss 和 Accuracy 曲线
trainer.plot_Accuracy()
trainer.plot_loss()
