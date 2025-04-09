from my_model.NeuralNetwork import NeuralNetwork
from load_data import load_data
import itertools 
import pandas as pd
from my_model.optimizer import SGD
from my_model.Trainer import Trainer
from my_model.evaluate import accuracy

X_train, y_train, X_val, y_val, X_test, y_test= load_data()
results = []
input = X_train.shape[1]
output = y_train.shape[1]

# 超参数范围
h1_sizes = [512, 256]
h2_sizes = [128, 64]
lr_list = [0.1, 0.001] 
reg_lambda_list = [1e-3, 1e-5]
    
# 所有超参数组合
param_grid = itertools.product(h1_sizes, h2_sizes, lr_list, reg_lambda_list)

for h1, h2, lr, reg in param_grid:
    print(f"Training with hidden1={h1}, hidden2={h2}, lr={lr:.5f}, reg={reg:.5f}")
    
    model = NeuralNetwork(input, h1, h2, output, activation='leaky_relu', reg_lambda=reg)
    optimizer = SGD(lr=lr, decay_rate=0.01)
    trainer = Trainer(model=model, optimizer=optimizer, batch_size=128, epochs=30)
    trainer.train(X_train, y_train, X_val, y_val)
    
    val_acc = accuracy(model, X_val, y_val)
    test_acc = accuracy(model, X_test, y_test)
    results.append((h1, h2, lr, reg, val_acc, test_acc))
    print(f"Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
# 保存结果
columns = ["Hidden Layer 1", "Hidden Layer 2", "Learning Rate", "L2 Regularization", "Validation Accuracy", "Test Accuracy"]
df = pd.DataFrame(results, columns=columns)
df.to_csv("hyperparam_results.csv", index=False)
print("\n结果已保存为 hyperparam_results.csv")