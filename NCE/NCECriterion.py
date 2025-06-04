import torch
from torch import nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch.nn.functional as F

eps = 1e-7
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs[:,1], targets.float(), reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs[:,1], targets.float(), reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return torch.mean(F_loss)


class NCECriterion(nn.Module):

    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, gpu_ids):
        bsz = x.shape[0]
        x = x.squeeze(-1)
        if len(x.shape) != 1:
            label = torch.zeros([bsz]).cuda(gpu_ids).long()
            
            loss = self.criterion(x, label)
        else:
            loss = self.criterion(x, torch.ones(bsz).cuda(gpu_ids))
        return loss

class NCESoftmaxLoss_smooth(nn.Module):
    """Softmax cross-entropy loss with label smoothing"""
    def __init__(self, num_classes, smoothing=0.1):
        super(NCESoftmaxLoss_smooth, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.criterion = nn.CrossEntropyLoss(reduction='mean')  # 设置平均损失

    def forward(self, x, gpu_ids):
        bsz = x.shape[0]
        x = x.squeeze(-1)
        
        # 创建平滑标签
        label = torch.zeros([bsz, self.num_classes]).cuda(gpu_ids)
        label.fill_(self.smoothing / (self.num_classes - 1))  # 将每个类别设置为平滑值
        label.scatter_(1, torch.zeros(bsz, dtype=torch.long).cuda(gpu_ids).unsqueeze(1), 1.0 - self.smoothing)  # 真实类别赋予高置信度

        # 计算损失
        loss = self.criterion(x, label)  # 这里使用的是平滑标签
        return loss
    
class ClusterLoss(nn.Module):
    
    def __init__(self, alpha, beta):
        super(ClusterLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.kmeans = KMeans(n_clusters=2, random_state=0)
    def forward(self, feat, label, gpu_ids):
        kmeans = self.kmeans.fit(feat.cpu().detach().numpy() )
        labels_pred = kmeans.labels_
        ari = adjusted_rand_score(label.cpu(), labels_pred)
        ari_loss = 1 - ari
        centers = kmeans.cluster_centers_
        wcss = 0
        for i in range(len(centers)):
            cluster_points = feat[label == i]
            wcss += np.sum((cluster_points - centers[i]) ** 2)

        # 计算数据集的总方差（最大可能的 WCSS）
        total_variance = np.sum((feat - np.mean(feat, axis=0)) ** 2)

        # 归一化 WCSS
        normalized_wcss = wcss / total_variance
                # Combine ARI loss and normalized WCSS
        combined_loss = ari_loss*self.alpha + normalized_wcss * self.beta
        
        # Convert combined_loss to a tensor with requires_grad=True
        loss_tensor = torch.tensor(combined_loss, dtype=torch.float32, requires_grad=True).cuda(gpu_ids)
        
        return loss_tensor