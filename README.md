# CIFAR-10 三层神经网络分类器（手工实现）

本项目使用纯 `NumPy` 实现了一个三层神经网络，对 CIFAR-10 图像数据集进行分类任务，支持模型训练、测试和超参数搜索，不依赖 PyTorch / TensorFlow 等自动微分框架。



## 项目结构

```bash
repo/
├── README.md                   # 项目说明
├── cifar10_data/               # 解压后的数据集
├── figs/                       # 训练过程和参数可视化图像
├── saved_models/               # 保存的训练模型

├── my_model/                   # 模型相关代码包
│   ├── __init__.py
│   ├── NeuralNetwork.py        # 网络结构定义
│   ├── Trainer.py              # 训练逻辑
│   ├── optimizer.py            # 优化器
│   ├── evaluate.py             # 评估函数（准确率、损失等）

├── test_train.py               # 训练入口
├── test_model.py               # 模型测试入口
├── search.py                   # 超参数搜索
├── weight_visualization.py     # 权重可视化脚本
├── hyperparam_results.csv      # 超参数搜索记录
└── train_log.txt               # 训练日志
```



### 训练模型

1. 打开 `test_train.py` 文件，自定义以下超参数：

   - `hidden_1`：第一隐藏层大小

   - `hidden_2`：第二隐藏层大小

   - `activation`：激活函数，支持 `'relu'` / `'sigmoid'` / `'leaky_relu'`

   - `reg_lambda`：L2 正则化系数（默认 0.001）

   - `lr`：学习率

   - `decay_rate`：学习率衰减率

   - `epochs`：训练轮数

   - `batch_size`：每批样本数

     

2. 运行 `train.py` 文件训练模型，训练好的模型权重文件`model.pkl`将自动保存并放置在`saved_models/`目录下。

   ```bash
   python train.py
   ```

   训练完毕后，将绘制训练集和验证集的 loss 曲线和验证集的 acccuracy 曲线，结果自动保存至 `figs/`目录下。
   
   

### 测试模型

1. 运行测试文件 `test_model.py`，将加载训练好的模型并输出在测试集上的准确率：

   ```bash
   python test_model.py
   ```


2. 如需测试其他已训练好的模型，请在 `load_model()` 中传入模型 `.pkl` 文件的路径：

   ```python
   model = NeuralNetwork.load_model('your_filepath')
   ```
   再运行测试文件 `test_model.py`：
   ```bash
   python test_model.py



### 超参数搜索

打开文件 `search.py`，填写需要搜索的超参数范围，并运行文件：

```bash
python search.py
```

最终将生成结果文件 `hyperparam_results.csv` 自动保存在当前目录下。



### 参数可视化

1. 运行文件 `weight_visualization.py` ，将可视化模型权重，结果自动保存至 `figs/`目录下。

   ```bash
   python weight_visualization.py
   ```

2. 如需可视化其他已训练好的模型权重，请在 `load_model()` 中传入模型 `.pkl` 文件的路径：

   ```bash
   model = NeuralNetwork.load_model('your_filepath')
   ```

   再运行文件 `weight_visualization.py`：

   ```bash
   python weight_visualization.py
   ```















































