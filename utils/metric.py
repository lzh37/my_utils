import torch

def metric(masks, target_masks):
    # mask, target_masks 均只含0,1
    masks_sum = torch.sum(masks)
    target_masks_sum = torch.sum(target_masks)
    inter_sum = torch.sum(masks * target_masks)
    iou = inter_sum / (target_masks_sum + masks_sum - inter_sum)
    return iou