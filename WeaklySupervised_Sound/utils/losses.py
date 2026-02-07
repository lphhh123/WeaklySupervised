import torch
import torch.nn as nn


class BCELossWithLabelSmoothing(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.bce = nn.BCELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce(inputs, targets)


class RobustBCELoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, epsilon: float = 1e-6):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.epsilon = float(epsilon)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        inputs = torch.clamp(inputs, min=self.epsilon, max=1.0 - self.epsilon)
        loss = -(targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        return loss.mean()
