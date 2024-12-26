import torch


@torch.no_grad()
def cal_sdf_acc(sdf: torch.Tensor, gt_occ: torch.Tensor) -> float:
    occ = torch.ones_like(sdf)
    occ[sdf > 0.0] = 0.0
    positive_acc_num = torch.where((occ > 0.5) & (gt_occ > 0.5))[0].shape[0]
    negative_acc_num = torch.where((occ < 0.5) & (gt_occ < 0.5))[0].shape[0]

    occ_num = 1
    for size in gt_occ.shape:
        occ_num *= size

    acc = (positive_acc_num + negative_acc_num) / occ_num
    return acc
