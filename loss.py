# File to store loss functions used to train UNet
import numpy as np
import torch


# return an array with the dice coeff per class
def get_dice_per_class(pred, target):
    smooth = 1e-3
    n_classes = pred.size(dim=1)
    dice = np.zeros(n_classes)

    iflat = torch.flatten(torch.swapaxes(pred, 0, 1), start_dim=1)
    tflat = torch.flatten(torch.swapaxes(target, 0, 1), start_dim=1)

    A_sum = iflat.sum(dim=1)
    B_sum = tflat.sum(dim=1)

    intersection = (iflat * tflat).sum(dim=1)
    num = (2 * intersection) + smooth
    denom = A_sum + B_sum + smooth

    dice = num / denom
    return dice


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    dice = get_dice_per_class(pred, target)
    dice = dice.mean(dim=0)
    #dice = torch.clamp(dice, 0, 1.0-epsilon)

    return  dice