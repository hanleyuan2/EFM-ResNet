import torch
import torch.nn as nn
import torch.nn.functional as F

class FEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEM, self).__init__()

        # 定义卷积层
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)

        # 批归一化层
        self.bn = nn.BatchNorm2d(out_channels)

        # 激活函数
        self.silu = nn.SiLU()

        # 池化层
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 合并卷积层（将池化后特征图合并）
        self.conv_merge = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, padding=1)

        # # 全连接层
        # self.fc = nn.Linear(out_channels * feature_map_size * feature_map_size, fc_out_dim)

        # Softmax层用于加权融合
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 第一步：3x3和5x5卷积
        R1 = self.silu(self.bn(self.conv3x3(x)))  # 使用3x3卷积
        R2 = self.silu(self.bn(self.conv5x5(x)))  # 使用5x5卷积
        print("R2", R2.shape)

        # 将R1和R2特征图进行加法融合
        Rm = R1 + R2
        print('Rm', Rm.shape)

        # 第二步：池化操作（最大池化和平均池化）
        Rn_max = self.max_pool(Rm)
        Rn_avg = self.avg_pool(Rm)
        print(Rn_avg.shape)
        # 将池化结果沿通道维度拼接
        Rn = torch.cat((Rn_max, Rn_avg), dim=1)
        print(Rn.shape)

        # 第三步：合并卷积
        Rp = self.conv_merge(Rn)
        print(Rp.shape)
        # Flatten并通过全连接层
        # M = self.fc(Rp.view(Rp.size(0), -1))  # Flatten the tensor for FC layer
        M = Rp.view(Rp.size(0), -1)  # Flatten the tensor for FC layer
        print(M.shape)

        # 第四步：Softmax进行加权融合
        gamma = self.softmax(M)  # Softmax用于生成权重系数

        # 根据生成的权重进行加权融合
        Rx = (R1 * gamma[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)) + \
             (R2 * gamma[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        return Rx

class EMA(nn.Module):
    def __init__(self, channels, factor=1):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class Bottleneck(nn.Module):      # 右侧的 residual block 结构（50-layer、101-layer、152-layer）
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):      # 三层卷积 Conv2d + Shutcuts
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.EMA = EMA(channels=planes, factor=planes)
        self.FEM = FEM(in_channels=planes, out_channels=planes)


        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)                  # conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)       # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)      # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)      # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)      # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out


def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def test():
    net = ResNet50()
    print(net)
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())


if __name__ == '__main__':
    test()
