import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 准备数据（自动下载手写数字数据集）
print("正在准备数据，第一次运行需要下载，请稍等...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 2. 搭建一个极其简单的神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.fc = nn.Linear(16 * 14 * 14, 10)  # 全连接层

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x


model = SimpleCNN()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 开始训练（这就是陈老师说的 Epoch）
print("开始训练...")
for epoch in range(2):  # 咱们只跑2轮，速度很快
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 300 == 299:  # 每300个批次打印一次进度
            print(f"Epoch {epoch + 1}, Batch {i + 1}, 当前误差(Loss): {running_loss / 300:.4f}")
            running_loss = 0.0

print("训练完成！恭喜你，你的AI已经认识手写数字了！")
#生成四张带标签的图像
import matplotlib.pyplot as plt
# 加上下面这两行
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

import matplotlib.pyplot as plt

print("正在生成图片预览...")
# 从测试集里拿 4 张图片出来看看
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
images, labels = next(iter(test_loader))

# 把矩阵数据还原成能显示的图片格式
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i in range(4):
    # 还原图片
    img = images[i].numpy().squeeze()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"标签: {labels[i].item()}")
    axes[i].axis('off')
model.eval()  # 切换到评估模式
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)  # 取分数最高的那个

for i in range(4):
    axes[i].set_title(f"真实:{labels[i].item()} 预测:{predicted[i].item()}")
plt.show() # 弹出窗口显示图片
