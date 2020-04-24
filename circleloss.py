import math
import torch 
import torch.nn as nn


class LabelCircleLossModel(nn.Module):
    def __init__(self, num_classes, m=0.35, gamma=30, feature_dim=192):
        super(LabelCircleLossModel, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, num_classes, requires_grad=True)).cuda()
        self.labels = torch.tensor([x for x in range(num_classes)]).cuda()
        self.classes = num_classes
        self.init_weights()
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        self.loss = nn.CrossEntropyLoss()
    def init_weights(self, pretrained=None):
        self.weight.data.normal_()

    def _forward_train(self, feat, label):
        normed_feat = torch.nn.functional.normalize(feat)
        normed_weight = torch.nn.functional.normalize(self.weight,dim=0)

        bs = label.size(0)
        mask = label.expand(self.classes, bs).t().eq(self.labels.expand(bs,self.classes)).float() 
        y_true = torch.zeros((bs,self.classes),device="cuda").scatter_(1,label.view(-1,1),1)
        y_pred = torch.mm(normed_feat,normed_weight)
        y_pred = y_pred.clamp(-1,1)
        sp = y_pred[mask == 1]
        sn = y_pred[mask == 0]

        alpha_p = (self.O_p - y_pred.detach()).clamp(min=0)
        alpha_n = (y_pred.detach() - self.O_n).clamp(min=0)

        y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
                    (1-y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
        loss = self.loss(y_pred,label)

        return loss, sp, sn

    def forward(self, input, label,  mode='train'):
            if mode == 'train':
                return self._forward_train(input, label)
            elif mode == 'val':
                raise KeyError

        
