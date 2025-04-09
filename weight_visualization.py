import numpy as np
import os
import matplotlib.pyplot as plt 
from my_model.NeuralNetwork import NeuralNetwork
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def visualize_top_W1(W1, img_shape=(32, 32, 3), top_k=16, filename='W1_top_visual.png'):
    os.makedirs('figs', exist_ok=True)  # 创建文件夹（若不存在）
    save_path = os.path.join('figs', filename)

    norms = np.linalg.norm(W1, axis=0)
    top_indices = np.argsort(norms)[-top_k:][::-1]  # 从大到小排列

    num_cols = int(np.sqrt(top_k))
    num_rows = (top_k + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))

    for i, idx in enumerate(top_indices):
        ax = axes[i // num_cols, i % num_cols]
        weight = W1[:, idx]
        img = weight.reshape(img_shape)

        # 转为灰度图
        img_gray = img.mean(axis=2)
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

        ax.matshow(img_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle(f"W1 中权重范数最大的前 {top_k} 个神经元", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.show()

def visualize_weight_heatmap(W, title="权重热力图", filename="heatmap.png"):
    os.makedirs('figs', exist_ok=True)  # 创建文件夹（若不存在）
    save_path = os.path.join('figs', filename)

    plt.figure(figsize=(10, 6))
    plt.imshow(W, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("下一层神经元")
    plt.ylabel("上一层神经元")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_weight_hist(weights, title="权重分布直方图", bins=50, filename="weight_hist.png"):
    os.makedirs('figs', exist_ok=True)  # 创建文件夹（若不存在）
    save_path = os.path.join('figs', filename)

    plt.figure(figsize=(6, 4))
    plt.hist(weights.flatten(), bins=bins, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("权重值")
    plt.ylabel("频数")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 加载训练好的模型
model = NeuralNetwork.load_model()

visualize_top_W1(model.W1)
visualize_weight_heatmap(model.W2, title="W2 权重热力图", filename="W2_heatmap.png")
visualize_weight_heatmap(model.W3, title="W3 权重热力图", filename="W3_heatmap.png")
plot_weight_hist(model.W2, title="W2 权重分布直方图", filename="W2_hist.png")
plot_weight_hist(model.W3, title="W3 权重分布直方图", filename="W3_hist.png")