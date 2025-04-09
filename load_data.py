import os
import pickle
import numpy as np

def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X = data_dict[b'data']  
        y = data_dict[b'labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
        return X, y

def load_cifar10(folder_path):
    X_train, y_train = [], []
    for i in range(1, 6):
        file = os.path.join(folder_path, f'data_batch_{i}')
        X, y = load_cifar_batch(file)
        X_train.append(X)
        y_train += y
    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)

    X_test, y_test = load_cifar_batch(os.path.join(folder_path, 'test_batch'))
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

def one_hot(y, num_classes=10):
    one_hot_labels = np.zeros((len(y), num_classes))
    one_hot_labels[np.arange(len(y)), y] = 1
    return one_hot_labels

def load_data():
    cifar_folder = './cifar10_data/cifar-10-batches-py'
    
    X_train, y_train, X_test, y_test = load_cifar10(cifar_folder)

    # 归一化
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 调整数据形状
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # one-hot 编码
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    val_size = 5000
    train_size = X_train.shape[0] - val_size
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    return X_train, y_train, X_val, y_val, X_test, y_test
