import torch
from torch import nn
from . import lossfunction, scaler, optimizer
from .option import opt
from .utils import dataset_mapping, dataloader, expected_caibration_error, calibrator_mapping, loss_mapping

# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'imagenet': 1000
}

class calibrator(nn.Module):
    def __init__(self, args):
        super(calibrator, self).__init__()
        torch.manual_seed(1)
        self.data = args.dataset
        self.n_class = dataset_num_classes[args.dataset]
        # Call the function to measure accuracy and expected calibration error.
        self.evaluator = expected_caibration_error()

        # Call the scaler for selected temperature based approach.
        self.scaler = getattr(scaler, calibrator_mapping(args.cal))(args)
        
        # Call the loss for learn.
        self.loss = getattr(lossfunction, loss_mapping('CE'))(args)

        # Call the optimizer for corresponding combination of scaler and loss.
        self.optim = args.optim
        self.optimizer = [getattr(optimizer, self.optim)(self.scaler, args)] if args.cal != 'ETS' else [
            getattr(optimizer, self.optim)(self.scaler.t, args), getattr(optimizer, self.optim)(self.scaler.w, args)
        ]
    
    def evaluate(self, x, y):
        # Evaluate the uncalibrated logits.
        uncalibrated_ece, uncalibrated_acc = self.evaluator(x, y)

        # Calibrate the logits.
        q = self.forward(x)

        # Evaluate the calibrated logits.
        calibrated_ece, calibrated_acc = self.evaluator(q, y)

    
    def forward(self, x):
        return self.scaler(x)
    
    def train(self, x, y):
        # Training for Parameterized Temperature Scaling
        if 'adam' in self.optim:
            for optimizers in self.optimizer:
                for optimizer, epochs in optimizers:
                    for _ in range(epochs):
                        optimizer.zero_grad()
                        q = self.forward(x)
                        loss = self.loss(q, y)
                        loss.backward()
                        optimizer.step()

        # Traninig for Temperature Scaling and Class-based Temperature Scaling
        else:
            for optimizers in self.optimizer:
                for optimizer in optimizers:
                    def trainer():
                        optimizer.zero_grad()
                        q = self.forward(x)
                        loss = self.loss(q, y)
                        loss.backward()

                        return loss

                    optimizer.step(trainer)
