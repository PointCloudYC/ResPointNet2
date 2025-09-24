import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logit, target, mask):
        """[summary]

        Args:
            logit ([type]): (B,C,N)
            target ([type]): (B,N)
            mask ([type]): (B,N)

        Returns:
            [type]: average loss value
        """
        # reduction = none to make similar shape as mask
        loss = F.cross_entropy(logit, target, reduction='none')
        loss *= mask
        mask_sum = mask.sum()
        
        # Handle edge case where mask is all zeros to prevent NaN
        if mask_sum == 0:
            return torch.tensor(0.0, device=logit.device, requires_grad=True)
        
        return loss.sum() / mask_sum
