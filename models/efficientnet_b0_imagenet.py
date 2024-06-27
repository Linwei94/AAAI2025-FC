# resnet with feature output
import torch
import torch.nn as nn
import torchvision


class EfficientNet_B0_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(EfficientNet_B0_ImageNet, self).__init__()
        self.model = torchvision.models.efficientnet_b0(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.classifier = self.model.classifier
        self.average_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x, return_feature=False):
        features = self.feature_extractor(x)
        features = self.average_pool(features)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
