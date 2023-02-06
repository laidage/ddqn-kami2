import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    # 神经网络
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.head = nn.Linear(3072, 290 * 4, dtype=torch.float64)

    def forward(self, x):
        # 向前函数，在nn.Module的__call__函数中被调用，不用显式调用
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))