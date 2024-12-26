import torch


@torch.no_grad()
def cal_occ_positive_percent(gt_occ: torch.Tensor) -> float:
    occ_shape = gt_occ.shape
    positive_occ_num = torch.where(gt_occ > 0.5)[0].shape[0]
    occ_num = 1
    for size in occ_shape:
        occ_num *= size

    positive_occ_percent = 1.0 * positive_occ_num / occ_num
    return positive_occ_percent

@torch.no_grad()
def cal_occ_acc(occ: torch.Tensor, gt_occ: torch.Tensor) -> float:
    positive_acc_num = torch.where((occ.sigmoid() > 0.5) & (gt_occ > 0.5))[0].shape[0]
    negative_acc_num = torch.where((occ.sigmoid() < 0.5) & (gt_occ < 0.5))[0].shape[0]

    occ_num = 1
    for size in gt_occ.shape:
        occ_num *= size

    acc = (positive_acc_num + negative_acc_num) / occ_num
    return acc
