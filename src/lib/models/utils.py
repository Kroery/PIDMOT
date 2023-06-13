from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torch.nn as nn
import numpy as np


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _gather_feat_avgregion(feat, ind, w, mask=None):
    device = feat.device
    b, n, dim = feat.shape
    # dim  = feat.size(2)
    # feat_region = torch.zeros([b, 500, dim, 9]).to(device=device, non_blocking=True)
    # ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), 9)
    # ind[:, :, 0] = torch.clamp(ind[:, :, 0] - w - 1, 0, 272*152)
    # ind[:, :, 1] = torch.clamp(ind[:, :, 1] - w, 0, 272*152)
    # ind[:, :, 2] = torch.clamp(ind[:, :, 2] - w + 1, 0, 272*152)
    # ind[:, :, 3] = torch.clamp(ind[:, :, 3] - 1, 0, 272*152)
    # ind[:, :, 5] = torch.clamp(ind[:, :, 5] + 1, 0, 272*152)
    # ind[:, :, 6] = torch.clamp(ind[:, :, 6] + w - 1, 0, 272*152)
    # ind[:, :, 7] = torch.clamp(ind[:, :, 7] + w, 0, 272*152)
    # ind[:, :, 8] = torch.clamp(ind[:, :, 8] + w + 1, 0, 272*152)

    # for i in range(ind.size(2)):
    #     # ind_temp = ind
    #     ind_temp = ind[:, :, 0].unsqueeze(2).expand(ind[:, :, 0].size(0), ind[:, :, 0].size(1), dim)
    #     feat_region[:, :, :, i] = feat.gather(1, ind_temp)
    #     # feat = feat.gather(1, ind_temp)
    ind_temp = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim, ind.size(2))
    feat_temp = feat.unsqueeze(3).expand(feat.size(0), feat.size(1), feat.size(2), 9)
    feat_temp = feat_temp.gather(1, ind_temp)
    # ind_temp = ind[:, :, 1].unsqueeze(2).expand(ind[:, :, 1].size(0), ind[:, :, 1].size(1), dim)
    # feat_region[:, :, :, 1] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 2].unsqueeze(2).expand(ind[:, :, 2].size(0), ind[:, :, 2].size(1), dim)
    # feat_region[:, :, :, 2] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 3].unsqueeze(2).expand(ind[:, :, 3].size(0), ind[:, :, 3].size(1), dim)
    # feat_region[:, :, :, 3] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 4].unsqueeze(2).expand(ind[:, :, 4].size(0), ind[:, :, 4].size(1), dim)
    # feat_region[:, :, :, 4] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 5].unsqueeze(2).expand(ind[:, :, 5].size(0), ind[:, :, 5].size(1), dim)
    # feat_region[:, :, :, 5] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 6].unsqueeze(2).expand(ind[:, :, 6].size(0), ind[:, :, 6].size(1), dim)
    # feat_region[:, :, :, 6] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 7].unsqueeze(2).expand(ind[:, :, 7].size(0), ind[:, :, 7].size(1), dim)
    # feat_region[:, :, :, 7] = feat.gather(1, ind_temp)
    # ind_temp = ind[:, :, 8].unsqueeze(2).expand(ind[:, :, 8].size(0), ind[:, :, 8].size(1), dim)
    # feat_region[:, :, :, 8] = feat.gather(1, ind_temp)

    # ind_temp = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # feat = feat.gather(1, ind_temp)
    # if mask is not None:
    #     mask = mask.unsqueeze(2).expand_as(feat)
    #     feat = feat[mask]
    #     feat = feat.view(-1, dim)
    feat = feat_temp
    # feat = feat_temp.mean(dim=3)
    return feat

def _gather_feat_avgregion_mean(feat, ind, w, mask=None):
    device = feat.device
    b, n, dim = feat.shape

    ind_temp = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim, ind.size(2))
    feat_temp = feat.unsqueeze(3).expand(feat.size(0), feat.size(1), feat.size(2), 9)
    feat_temp = feat_temp.gather(1, ind_temp)

    feat = feat_temp.mean(dim=3)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _tranpose_and_gather_feat_avgregion(feat, ind, w):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat_avgregion(feat, ind, w)
    return feat

def _tranpose_and_gather_feat_avgregion_mean(feat, ind, w):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat_avgregion_mean(feat, ind, w)
    return feat

def _tranpose_and_gather_feat_region(feat, region, mask):
    device = feat.device
    b,d,h,w = feat.shape
    _,n,_ = region.shape
    # region[:, :, 0] = (region[:, :, 0] + region[:, :, 2]) / 2
    # region[:, :, 1] = (region[:, :, 1] + region[:, :, 3]) / 2
    # region[:, :, 2] = region[:, :, 0] + 1
    # region[:, :, 3] = region[:, :, 1] + 1

    # id_feat = np.zeros((b, n, d, 1, 1), dtype=np.float32)
    id_feat = torch.zeros([b, n, d, 1, 1])
    # pl = nn.AdaptiveAvgPool2d((1,1))
    # feat = feat.permute(0, 2, 3, 1).contiguous()
    # feat = feat.view(feat.size(0), -1, feat.size(3))
    # feat = _gather_feat(feat, region)
    for i in range(b):
        mask_i = mask[i]
        input = feat[i].unsqueeze(0).cpu()
        # boxes = region[i][mask_i].cpu()
        boxes = region[i].cpu()
        id_feat[i] = torchvision.ops.roi_pool(input, [boxes], (1,1))

    id_feat = torch.tensor(id_feat).to(device=device, non_blocking=True)

    # id_feat_flatten = pl(id_feat.view(-1, d, 7, 7))

    # id_feat = id_feat.view(id_feat.size(0), id_feat.size(1), -1)
    #
    # fc = nn.Linear(6272, 64).cuda()
    # id_feat_flatten = fc(id_feat.view(-1, id_feat.size(-1))).view(b, n, 64)

    # fc = nn.Conv2d(id_feat.size(-1), 64, 1, bias=False).cuda()
    # id_feat_flatten = fc(id_feat.unsqueeze(0).permute(0,3,1,2))

    # return id_feat_flatten.permute(0,2,3,1).squeeze(0)
    return id_feat.squeeze(-1).squeeze(-1)
    # return id_feat_flatten.view(b, n, d, 1, 1).squeeze(-1).squeeze(-1)


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)