# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def _load_class_hierarchy(hierarchy_file):

    assert os.path.exists(hierarchy_file), "File {} dose not exist.".format(hierarchy_file)
    
    print('Loading', hierarchy_file)
    hierarchy_data = json.load(open(hierarchy_file, 'r'))
    is_childs = torch.Tensor(hierarchy_data['is_childs']).float()  # (C + 1) x (C + 1), the last row / column is the background
    is_parents = torch.Tensor(hierarchy_data['is_parents']).float()  # (C + 1) x (C + 1)
    
    assert (is_childs * is_parents).sum() == 0 and \
        (is_childs[-1, :].sum() + is_childs[:, -1].sum() + is_parents[-1, :].sum() + is_parents[:, -1].sum()) == 0
    
    return is_parents, is_childs


@LOSSES.register_module()
class HierarchyLoss(nn.Module):
    # case 1: no special treatment for hierarchy label
    #     set_pos_parents=False
    #     ignore_children=False
    # case 2: ignore parents and children in oid label space
    #     set_pos_parents=False
    #     ignore_children=True
    #     pos_parents="oid"
    #     ignore="child++"
    # case 3: ignore parents and children in rvc label space
    #     set_pos_parents=False
    #     ignore_children=True
    #     pos_parents="rvc"
    #     ignore="child++"
    # case 4: set parents as positive in oid label space, ignore children in oid label space
    #     set_pos_parents=True
    #     ignore_children=True
    #     pos_parents="oid"
    #     ignore="child"
    # case 5: set parents as positive in rvc label space, ignore children in rvc label space
    #     set_pos_parents=True
    #     ignore_children=True
    #     pos_parents="rvc"
    #     ignore="child"
    # case 6: set parents as positive in oid label space, ignore children in rvc label space, ignore parents in rvc label space without oid
    #     set_pos_parents=True
    #     ignore_children=True
    #     pos_parents="oid"
    #     ignore="child+"

    def __init__(self,
                 set_pos_parents=True,
                 ignore_children=True,
                 pos_parents="oid",
                 ignore="child", 
                 hierarchy_oid_file="./label_spaces/hierarchy_oid.json",
                 hierarchy_rvc_file="./label_spaces/hierarchy_rvc.json",
                 loss_weight=1.0):
        super(HierarchyLoss, self).__init__()
        self.set_pos_parents = set_pos_parents
        self.ignore_children = ignore_children
        self.hierarchy_oid_file = hierarchy_oid_file
        self.hierarchy_rvc_file = hierarchy_rvc_file
        self.loss_weight = loss_weight
        ip_oid, ic_oid = _load_class_hierarchy(hierarchy_oid_file)
        ip_rvc, ic_rvc = _load_class_hierarchy(hierarchy_rvc_file)

        if pos_parents == "oid":
            is_parents = ip_oid
            is_children = ic_oid
        elif pos_parents == "rvc":
            is_parents = ip_rvc
            is_children = ic_rvc
        else:
            raise ValueError
        
        if ignore == "child":
            hierarchy_weight = 1 - is_children
        elif ignore == "child+":
            hierarchy_weight = 1 - (ip_rvc - ip_oid + ic_rvc)
        elif ignore == "child++":
            hierarchy_weight = 1 - (is_parents + is_children)
        else:
            raise ValueError

        self.is_parents = is_parents
        self.hierarchy_weight = hierarchy_weight
        
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none'), "reduction should be none"

        if pred.numel() == 0:
            return pred.new_zeros([1])[0]  # This is more robust than .sum() * 0.

        B = pred.shape[0]
        C = pred.shape[1] - 1

        target_refined = pred.new_zeros(B, C + 1).detach()
        target_refined[range(len(target)), target] = 1  # B x (C + 1)
        
        # set all parents to positive
        if self.set_pos_parents:
            is_parents = self.is_parents.to(target_refined.device).detach()  # (C + 1) x (C + 1)
            target_refined = torch.mm(target_refined, is_parents) + target_refined  # B x (C + 1)

        cls_loss = F.binary_cross_entropy_with_logits(pred, target_refined, reduction='none')  # B x (C + 1)

        # ignore all childern
        if self.ignore_children:
            hierarchy_w = self.hierarchy_weight.to(target_refined.device).detach()  # (C + 1) x (C + 1)
            hierarchy_w = hierarchy_w[target]  # B x (C + 1)
        else:
            hierarchy_w = target_refined.new_ones(B, C + 1).detach()

        cls_loss = torch.sum(cls_loss * hierarchy_w) / B
        
        return self.loss_weight * cls_loss