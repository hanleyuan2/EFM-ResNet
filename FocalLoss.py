import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        logits: 模型输出的未经过softmax的原始分数, 形状为 (batch_size, num_classes)
        labels: 真实标签, 形状为 (batch_size)
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, accuracies=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.accuracies = accuracies

    def forward(self, logits, labels):
        """
        logits: 模型输出的未经过softmax的原始分数, 形状为 (batch_size, num_classes)
        labels: 真实标签, 形状为 (batch_size)
        """
        focal_loss = FocalLoss(self.alpha, self.gamma)(logits, labels)
        if self.accuracies is not None:
            # 假设accuracies是一个形状为 (num_classes,) 的张量
            a_i = self.accuracies[labels]
            denominator = torch.sum(1 / self.accuracies)
            weighted_loss = (1 / a_i) / denominator * focal_loss
            return weighted_loss.mean()
        else:
            return focal_loss


if __name__ =='__main__':
    # 示例使用
    batch_size = 32
    num_classes = 14
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 测试焦点损失
    focal_loss_func = FocalLoss()
    loss_focal = focal_loss_func(logits, labels)
    print("Focal Loss:", loss_focal)

    # 假设上一轮各类别的准确率
    accuracies = torch.rand(num_classes)
    weighted_focal_loss_func = WeightedFocalLoss(accuracies=accuracies)
    loss_weighted_focal = weighted_focal_loss_func(logits, labels)
    print("Weighted Focal Loss:", loss_weighted_focal)









