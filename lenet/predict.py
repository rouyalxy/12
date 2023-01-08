import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transforms = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'car', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth')) # 接收字典对象

img = Image.open('th.jpg')
img = transforms(img)  # [C,H,W]
img = torch.unsqueeze(img, dim=0) # nn支持小批量处理，所以要增加一个batch维度 [N, C, H, W]

with torch.no_grad(): # 预测不需要求梯度
    outputs = net(img)
    # predict = torch.max(outputs, dim=1)[1].data.numpy()
    predict = torch.softmax(outputs, dim=1)  # dim=1是在channel这个维度上进行处理（最后一层输出为10个channel）
# print(classes[int(predict)])
print(predict)  # 生成概率分布
