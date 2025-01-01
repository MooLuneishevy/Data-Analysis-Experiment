# 导入必要的库
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# 设定超参数
n_epochs = 3  # 训练模型的epoch数
batch_size_train = 64  # 训练批大小
batch_size_test = 1000  # 测试批大小
learning_rate = 0.01  # 学习率
momentum = 0.5  # 使用动量优化器
log_interval = 10  # 记录训练损失的间隔
random_seed = 1  # 随机种子设定
torch.manual_seed(random_seed)
# 加载MNIST训练数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
# 加载MNIST测试数据
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
# 获取训练数据进行可视化
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# 打印真实标签
# print(example_targets)
# 打印训练数据的形状
# print(example_data.shape)
# 绘制前六个训练图像
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("label:{}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义网络层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 卷积
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 卷积
        self.conv2_drop = nn.Dropout2d()  # Dropout层
        self.fc1 = nn.Linear(320, 50)  # 全连接层
        self.fc2 = nn.Linear(50, 10)  # 全连接层
    def forward(self, x):
        # 定义网络的前向传播
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 卷积层1使用ReLU激活函数和最大池化
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # 卷积层2使用ReLU激活函数Dropout和最大池化
        x = x.view(-1, 320)  # 展开特征图
        x = F.relu(self.fc1(x))  # 全连接层1使用ReLU激活函数
        x = F.dropout(x, training=self.training)  # 训练时使用Dropout层
        x = self.fc2(x)  # 全连接层2
        return F.log_softmax(x, dim=1)  # 使用对数Softmax激活函数 进行多分类
# 初始化神经网络
network = Net()
# 定义优化器 使用动量
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# 以列表存储训练损失 测试损失的列表
train_losses = []
train_counter = []
test_losses = []
test_losses.append(None)
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs+1)]
# 网络训练函数
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        output = network(data)
        # 计算损失
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 记录训练损失
        if batch_idx % log_interval == 0:
            print('Epoch:{} [{}/{} ({:.0f}%)]\	Loss:{:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            # 保存模型和优化器状态字典
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
# 网络测试函数
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # 前向传播
            output = network(data)
            # 计算损失
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 计算准确率
            pred = output.data.max(1, keepdim=True)[1]  # 获取对数概率最大的索引
            correct += pred.eq(target.data.view_as(pred)).sum()
    # 计算平均测试损失
    test_loss /= len(test_loader.dataset)
    # 记录测试损失和准确率
    print('\
Test Set:Average Loss:{:.4f} Accuracy:{}/{} ({:.0f}%)\
'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # 存储测试损失
    test_losses.append(test_loss)
# 网络训练一个epoch
train(1)
# 测试网络
test()
# 网络训练剩余epoch
for epoch in range(2, n_epochs + 1):
    train(epoch)
    test()
# 绘制训练和测试损失图像
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Training Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Number of training samples seen')
plt.ylabel('Negative log likelihood loss')
# 加载测试数据 进行可视化
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# 对测试数据进行预测
with torch.no_grad():
    output = network(example_data)
# 绘制预测结果
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction:{}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
# 初始化新的网络和优化器 用于持续训练
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# 加载模型和优化器状态字典
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
# 继续训练模型
for i in range(4, 9):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()
# 作图可视化网络训练过程
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()