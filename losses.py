import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, weight_mse=1.0, weight_l1=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_l1 = weight_l1
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, input, target):
        mse = self.mse_loss(input, target)
        l1 = self.l1_loss(input, target)
        combined_loss = (self.weight_mse * mse) + (self.weight_l1 * l1)
        return combined_loss