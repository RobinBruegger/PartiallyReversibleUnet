import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch

def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def bratsDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = diceLoss(wt, wtMask, nonSquared=nonSquared)
    tcLoss = diceLoss(tc, tcMask, nonSquared=nonSquared)
    etLoss = diceLoss(et, etMask, nonSquared=nonSquared)
    return (wtLoss + tcLoss + etLoss) / 5

def bratsDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss


def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getWTMask(labels):
    return (labels != 0).float()

def getTCMask(labels):
    return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

def getETMask(labels):
    return (labels == 4).float()
