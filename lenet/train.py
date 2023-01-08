import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model import LeNet


'''
torch.nn只支持小批量处理(mini-batches）。整个torch.nn包只支持小批量样本的输入，不支持单个样本的输入。

比如，nn.Conv2d 接受一个4维的张量，即nSamples x nChannels x Height x Width

如果是一个单独的样本，只需要使用input.unsqueeze(0)来添加一个“假的”批大小维度。

'''


transform = transforms.Compose([  # 传入列表，并实例化Compose。再调用transform后直接调用__call__()属性
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='../data', train=True,
                                         download=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=36,
                          shuffle=True, num_workers=0)  # num_workers=0 win系统下建议设为0，否则可能会报错

test_set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=640,
                         shuffle=False, num_workers=0)

test_data_iter = iter(test_loader)  # 把数据集生成迭代器
test_image, test_label = test_data_iter.__next__()

classes = ('plane', 'car', 'bird', 'car', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# print(test_label)
# def show(img):
#     img = img/2+0.5 # 反标准化
#     npimg = img.numpy()  # 将tensor->numpy格式
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 将tensor格式的输入转为正常图片显示的形式(C,H,W)->(H,W,C)
#     plt.show()
# #
# # print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# #
#
# show(torchvision.utils.make_grid(test_image))

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # 定义损失函数，包含了softmax，不用在输出加softmax
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器，优化所有可训练的参数
for epoch in range(5):

    running_loss = 0.0  # 叠加训练过程中的损失
    for step, data in enumerate(train_loader, start=0):  # 遍历训练集样本
        inputs, labels = data

        optimizer.zero_grad()  # 对参数梯度清零，不清零梯度会累加
        outputs = net(inputs)
        loss = loss_function(outputs, labels)  # 网络预测值与真实标签
        loss.backward()  # 方向传播
        optimizer.step()  # 参数更新
        # print(net.conv1.bias.grad)  # 查看偏执的loss
        # print(loss.grad_fn()) # 返回loss类
        # print(loss.grad_fn.next_functions[0][0])  # next_functions[0][0]下一步loss
        # print statistics
        running_loss += loss.item()  # tensor取数值

        if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():  # 该模块下，requires_grad都设置为False，不再计算梯度
                outputs = net(test_image)  # [batch, 10]
                # print(outputs)
                # print(outputs.size())
                predict_y = torch.max(outputs, dim=1)[1]  # dim=0是batch，[1]是取出最大值所在的索引，在通过索引找到预测标签。用argmax()更方便
                accuray = (predict_y == test_label).sum().item() / test_label.size(0)  # 预测对样本个数/总的个数
                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuray))
                running_loss = 0.0

print('Finished Traning')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)  # 将网络参数保存

# params = list(net.parameters())  # 一个模型可学习的参数net.parameters()
# print(len(params))
# print(params[0].size())  # conv1's .weight