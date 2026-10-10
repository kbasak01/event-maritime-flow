# ruff: noqa
# Vendored from uzh-rpg/E-RAFT (https://github.com/uzh-rpg/E-RAFT), commit c58ce0524ea0ebfa9849991caafb547f44fe9bfd,
# file `model/utils.py`, for INFERENCE ONLY (reference model R1). E-RAFT is MIT licensed:
# "Copyright (c) 2021 Robotics and Perception Group". The E-RAFT network is derived from RAFT
# (princeton-vl/RAFT, BSD-3-Clause, "Copyright (c) 2020, princeton-vl"). Both licence texts are
# reproduced in full in LICENSE next to this file; docs/LICENSES.md has the entry.
#
# Modifications: none.

import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
