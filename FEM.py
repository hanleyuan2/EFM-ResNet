import torch
import torch.nn as nn


class FEM(nn.Module):
    def __init__(self, in_channels, out_channels, feature_map_size, fc_out_dim):
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

        # 全连接层
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
# 示例：测试模块
if __name__ == "__main__":
    x = torch.randn(32, 3, 64, 64)  # 假设输入特征图尺寸为 56x56，通道数为 64
    module = FEM(in_channels=3, out_channels=3, feature_map_size=64, fc_out_dim=10)
    y = module(x)
    print("输出形状：", y.shape)