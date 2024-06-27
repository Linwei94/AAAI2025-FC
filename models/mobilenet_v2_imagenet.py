# resnet with feature output
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class MobileNet_V2_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(MobileNet_V2_ImageNet, self).__init__()
        self.model = torchvision.models.mobilenet_v2(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.classifier = self.model.classifier
    def forward(self, x, return_feature=False):
        features = self.feature_extractor(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
