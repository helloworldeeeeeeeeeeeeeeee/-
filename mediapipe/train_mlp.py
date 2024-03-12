import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from data import CustomDataset
from model import MLP





# 加载数据
train_folder = '../train_npy'  # 替换为包含 train_npy 和 test_npy 的文件夹路径
train_dataset = CustomDataset(train_folder)  # 自定义数据集类，加载你的数据

test_folder = '../test_npy'  # 替换为包含 train_npy 和 test_npy 的文件夹路径
test_dataset = CustomDataset(test_folder)  # 自定义数据集类，加载你的数据

# # 划分训练集和测试集
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 设置模型和优化器
input_size = 18 * 75 * 2  # 根据你的关键点数据维度进行调整
# hidden_size = 128
num_classes = 226
hidden_sizes = [512, 256, 128, 64]
model = MLP(input_size, hidden_sizes, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def test(model):
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy
# 训练模型
num_epochs = 1000
best_acc = 0
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    accuracy = test(model)
    if accuracy>best_acc:
        best_acc = accuracy
        print('epoch: ',epoch,'best_acc: ',best_acc)

