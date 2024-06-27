# resnet with feature output
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Wide_ResNet_ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super(Wide_ResNet_ImageNet, self).__init__()
        self.model = torchvision.models.wide_resnet50_2(**kwargs)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x, return_feature=False):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
    
    def classifier(self, x):
        return self.model.fc(x)
