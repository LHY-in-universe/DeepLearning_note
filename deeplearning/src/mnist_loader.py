import numpy as np
from tensorflow.keras.datasets import mnist

def load_data_wrapper():
    """使用Keras内置的MNIST加载器"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 预处理数据
    train_images = train_images.reshape((60000, 784, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 784, 1)).astype('float32') / 255

    # 分割验证集
    val_images = train_images[50000:]
    val_labels = train_labels[50000:]
    train_images = train_images[:50000]
    train_labels = train_labels[:50000]

    # 转换为one-hot
    training_results = [vectorized_result(y) for y in train_labels]
    training_data = list(zip(train_images, training_results))
    validation_data = list(zip(val_images, val_labels))
    test_data = list(zip(test_images, test_labels))

    return training_data, validation_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
