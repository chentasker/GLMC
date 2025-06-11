import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, coarse_to_fine=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
        if not coarse_to_fine:
            coarse_to_fine = {i:[i] for i in range(self.num_classes)}
        self.coarse_to_fine = coarse_to_fine
        self.all_classes = {i: [self.coarse_to_fine[j][i] for j in self.coarse_to_fine.keys()] for i in range(len(self.coarse_to_fine[0]))}
        

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        #classes = torch.arange(self.num_classes).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = torch.zeros_like(labels)
        for i in self.all_classes.keys():
            classes = self.all_classes[i]
            if self.use_gpu: classes = classes.cuda()
            mask = mask.logical_or(labels.eq(classes.expand(batch_size, self.num_classes)))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
