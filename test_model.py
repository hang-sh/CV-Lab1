from my_model.NeuralNetwork import NeuralNetwork
from load_data import load_data
from my_model.evaluate import accuracy
import datetime

# 导入数据
_, _, X_val, y_val, X_test, y_test = load_data()

# 加载训练好的模型进行测试
model = NeuralNetwork.load_model()
test_accuracy = accuracy(model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 日志内容
log_message = f"[{current_time}] Test Accuracy: {test_accuracy:.4f}\n"
log_message += "Model Parameters:\n"
log_message += f"input: {model.input}, output: {model.output}\n"
log_message += f"h1: {model.hidden_1}, h2: {model.hidden_2}\n"
log_message += f"activation: {model.activation}\n"
log_message += f"reg_lambda: {model.reg_lambda}\n"

# 将日志写入文件
with open("train_log.txt", "a") as log_file:
    log_file.write(log_message)
