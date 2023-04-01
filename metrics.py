import torch

def filter_by_mask(pred, mask):
    # filter out imgs that mask is all 0
    # pred: N,C,H,W
    # mask: N,C,H,W

    # sum mask over all c, x, y
    mask_sum = torch.sum(mask, dim=(1, 2, 3)) # N

    # filter out imgs that mask is all 0
    mask_sum = mask_sum > 0 # N

    pred = pred[mask_sum] # N',C,H,W
    mask = mask[mask_sum] # N',C,H,W

    return pred, mask


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x
    
def iou(pr, gt, eps=1e-7, threshold=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

def dice(pr, gt, eps=1e-7, threshold=None):
    """Calculate Dice cofficient between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: dice score
    """

    pr = _threshold(pr, threshold=threshold)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (2 * intersection + eps) / (union + intersection)
