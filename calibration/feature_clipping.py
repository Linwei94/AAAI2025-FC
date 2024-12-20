'''
Code to perform feature clipping. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from metrics.metrics import ECELoss
    
# implemented as a post hoc calibrator
class FeatureClippingCalibrator(nn.Module):
    def __init__(self, model, cross_validate='ece'):
        super(FeatureClippingCalibrator, self).__init__()
        self.cross_validate = cross_validate
        self.feature_clip = float("inf")
        self.ece_criterion = ECELoss().cuda()
        self.nll_criterion = nn.CrossEntropyLoss().cuda()
        self.model = model
        self.classifier = self.model.classifier

    def get_feature_clip(self):
        return self.feature_clip
    
    def set_feature_clip(self, features_val, logits_val, labels_val):
        nll_val_opt = float("inf")
        ece_val_opt = float("inf")
        C_opt_nll = float("inf")
        C_opt_ece = float("inf")
        self.feature_clip = float("inf")
        
        before_clipping_acc = (F.softmax(logits_val, dim=1).argmax(dim=1) == labels_val).float().mean().item()
        
        C = 0.01
        for _ in range(2000):
            logits_after_clipping = self.classifier(self.feature_clipping(features_val, C))
            after_clipping_nll = self.nll_criterion(logits_after_clipping, labels_val).item()
            after_clipping_ece = self.ece_criterion(logits_after_clipping, labels_val).item()
            after_clipping_acc = (F.softmax(logits_after_clipping, dim=1).argmax(dim=1) == labels_val).float().mean().item()
            if (after_clipping_nll < nll_val_opt) and (after_clipping_acc > before_clipping_acc*0.99):
                C_opt_nll = C
                nll_val_opt = after_clipping_nll

            if (after_clipping_ece < ece_val_opt) and (after_clipping_acc > before_clipping_acc*0.99):
                C_opt_ece = C
                ece_val_opt = after_clipping_ece

            C += 0.01

        if self.cross_validate == 'ece':
            self.feature_clip = C_opt_ece
        elif self.cross_validate == 'nll':
            self.feature_clip = C_opt_nll

        return self.feature_clip
    
    def feature_clipping(self, features, c=None):
        """
        Perform feature clipping on logits
        """

        return torch.clamp(features, min=-c, max=c)
    

    def forward(self, features, c=None):
        return self.classifier(self.feature_clipping(features, c))
