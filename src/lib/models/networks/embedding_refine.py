import warnings
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
# from mmcv.cnn import ConvModule
from einops import rearrange, repeat
from itertools import repeat as rp

from models.utils import _tranpose_and_gather_feat, _tranpose_and_gather_feat_avgregion
from models.decode import mot_decode

import time
import numpy as np


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x, class_token=False):
        # x = tensor_list.tensors
        # x: b,d,h,w
        num_feats = x.shape[1]
        num_pos_feats = num_feats // 2

        x_embed = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        x_embed = torch.Tensor(x_embed).to(x.device)

        # mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device).to(torch.bool)
        # batch = mask.shape[0]
        # assert mask is not None
        # not_mask = mask
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # if self.normalize:
        #     eps = 1e-5
        #     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        #     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)

        pos_x = x_embed[:, None] / dim_t
        # pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        # pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, h*w, d
        '''
        # if class_token:
        #     # pos = torch.cat((self.class_token_pos.repeat(batch, 1, 1), pos), dim=1)
        #     # pos = torch.cat((torch.mean(pos, dim=1, keepdim=True), pos), dim=1)
        #     pos = torch.cat((torch.zeros(batch, 1, pos.shape[2], dtype=pos.dtype, device=pos.device), pos), dim=1)
        return pos_x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask=None,
                     memory_mask=None,
                     tgt_key_padding_mask=None,
                     memory_key_padding_mask=None,
                     pos=None,
                     query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask=None,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        # q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask=None,
                     memory_mask=None,
                     tgt_key_padding_mask=None,
                     memory_key_padding_mask=None,
                     pos=None,
                     query_pos=None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask=None,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class EmbedRefine_hmvis(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(EmbedRefine_hmvis, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.cascade_num = 4
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        # self.trca = nn.ModuleList()
        # for _ in range(self.cascade_num):
        #     self.trca.append(TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
        #                                            dropout, activation, normalize_before))

        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size

    def to_patchq(self, x, ind, mask):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # out: [hw,B,d]->[1,66,128]

        x = _tranpose_and_gather_feat(x, ind)  # [3,500,128]
        x = x[mask > 0].contiguous()  # [66,128]
        q_patch = x.unsqueeze(0)
        return q_patch

    def to_patchk(self, x, ind_3x3, mask):
        # x: id feature [B,128,152,272]
        # ind_3x3: center [B,500,9]
        # out: [hw,B,d]->[9,66,128]

        x = _tranpose_and_gather_feat_avgregion(x, ind_3x3, x.size(-1))  # [3,500,128,9]
        x = x[mask > 0].contiguous()  # [66,128,9]
        k_patch = x.permute(2, 0, 1)
        return k_patch

    def forward(self, x, hm=None, wh=None, reg=None, ind=None, ind_3x3=None, mask=None, vis=None):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # ind_3x3: center_region [B,500,9]
        # mask: reg_mask [B,500]

        # B = len(ind.nonzero())

        # if ind is not None:
        #     q_patch = self.to_patchq(x, ind, mask)    # [1,66,128]
        #     k_patch = self.to_patchk(x, ind_3x3, mask)   # [9,66,128]
        # else:

        b, d, h, w = x.shape

        # start = time.time()
        hm = hm * vis
        dets, inds = mot_decode(hm, wh, reg=reg)

        inds_copy = copy.deepcopy(inds)
        w = x.size(-1)
        inds0 = torch.clamp(inds - w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds1 = torch.clamp(inds - w, 0, 272 * 152 - 1).unsqueeze(2)
        inds2 = torch.clamp(inds - w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds3 = torch.clamp(inds - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds4 = torch.clamp(inds, 0, 272 * 152 - 1).unsqueeze(2)
        inds5 = torch.clamp(inds + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds6 = torch.clamp(inds + w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds7 = torch.clamp(inds + w, 0, 272 * 152 - 1).unsqueeze(2)
        inds8 = torch.clamp(inds + w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds = torch.cat((inds0, inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), dim=2)

        q_patch = _tranpose_and_gather_feat(x, inds_copy)  # [1,100,128]
        q_patch = q_patch.view(-1, d).unsqueeze(0)
        k_patch = _tranpose_and_gather_feat_avgregion(x, inds, w)  # [1,100,128,9]
        k_patch = k_patch.view(-1, d, 9).permute(2, 0, 1).contiguous()  # [9,100,128]

        detections = dets[:, :, :4].int()
        b, n, _ = detections.shape

        # vis_pos = self.pose_encoding(x)
        vis = _tranpose_and_gather_feat(vis, inds_copy)
        vis = (vis * 10).long().view(-1, 1).contiguous()
        vis = vis.unsqueeze(2).expand(vis.size(0), vis.size(1), d)
        # vis_pos = vis_pos.unsqueeze(0).expand(b * n, vis_pos.size(0), vis_pos.size(1))
        # vis_pos = vis_pos.gather(1, vis).permute(1, 0, 2)

        # q_patch = q_patch + vis_pos
        out = self.tr_decoder(tgt=q_patch, memory=k_patch)
        out = out.squeeze(0).view(b, 500, d).contiguous()

        # for i in range(self.cascade_num):
        #     out = self.trca[i](tgt=q_patch, memory=k_patch)
        #     q_patch = out

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(3))
        for i in range(b):
            x[i, inds_copy[i], :] = out[i,:,:]

        x = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        # out = out.permute(1, 2, 0).reshape(b, d, h, w)
        # end = time.time()
        # print(end-start)
        return x


class EmbedRefine_hmvis_saca(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(EmbedRefine_hmvis_saca, self).__init__()
        self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.trfu = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        self.cascade_num = 4
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.trca = nn.ModuleList()
        # for _ in range(self.cascade_num):
        #     self.trca.append(TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
        #                                            dropout, activation, normalize_before))

        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size

    def to_patchq(self, x, ind, mask):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # out: [hw,B,d]->[1,66,128]

        x = _tranpose_and_gather_feat(x, ind)  # [3,500,128]
        x = x[mask > 0].contiguous()  # [66,128]
        q_patch = x.unsqueeze(0)
        return q_patch

    def to_patchk(self, x, ind_3x3, mask):
        # x: id feature [B,128,152,272]
        # ind_3x3: center [B,500,9]
        # out: [hw,B,d]->[9,66,128]

        x = _tranpose_and_gather_feat_avgregion(x, ind_3x3, x.size(-1))  # [3,500,128,9]
        x = x[mask > 0].contiguous()  # [66,128,9]
        k_patch = x.permute(2, 0, 1)
        return k_patch

    def forward(self, x, hm=None, wh=None, reg=None, ind=None, ind_3x3=None, mask=None, vis=None):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # ind_3x3: center_region [B,500,9]
        # mask: reg_mask [B,500]

        # B = len(ind.nonzero())

        # if ind is not None:
        #     q_patch = self.to_patchq(x, ind, mask)    # [1,66,128]
        #     k_patch = self.to_patchk(x, ind_3x3, mask)   # [9,66,128]
        # else:

        b, d, h, w = x.shape

        # start = time.time()
        hm = hm * vis
        dets, inds = mot_decode(hm, wh, reg=reg)

        inds_copy = copy.deepcopy(inds)
        w = x.size(-1)
        inds0 = torch.clamp(inds - w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds1 = torch.clamp(inds - w, 0, 272 * 152 - 1).unsqueeze(2)
        inds2 = torch.clamp(inds - w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds3 = torch.clamp(inds - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds4 = torch.clamp(inds, 0, 272 * 152 - 1).unsqueeze(2)
        inds5 = torch.clamp(inds + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds6 = torch.clamp(inds + w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds7 = torch.clamp(inds + w, 0, 272 * 152 - 1).unsqueeze(2)
        inds8 = torch.clamp(inds + w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds = torch.cat((inds0, inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), dim=2)

        q_patch = _tranpose_and_gather_feat(x, inds_copy)  # [1,100,128]
        q_patch = q_patch.view(-1, d).unsqueeze(0)
        k_patch = _tranpose_and_gather_feat_avgregion(x, inds, w)  # [1,100,128,9]
        k_patch = k_patch.view(-1, d, 9).permute(2, 0, 1).contiguous()  # [9,100,128]

        detections = dets[:, :, :4].int()
        b, n, _ = detections.shape

        vis_pos = self.pose_encoding(x)
        vis = _tranpose_and_gather_feat(vis, inds_copy)
        vis = (vis * 10).long().view(-1, 1).contiguous()
        vis = vis.unsqueeze(2).expand(vis.size(0), vis.size(1), d)
        vis_pos = vis_pos.unsqueeze(0).expand(b * n, vis_pos.size(0), vis_pos.size(1))
        vis_pos = vis_pos.gather(1, vis).permute(1, 0, 2)

        # q_patch = q_patch + vis_pos
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b * n).permute(1, 0, 2).contiguous()
        encoder_patch = torch.cat((q_patch, k_patch), dim=0)

        en_out = self.tr_encoder(src=encoder_patch)[0]
        en_out = en_out.unsqueeze(0)

        de_out = self.tr_decoder(tgt=q_patch, memory=k_patch)

        fu_patch = torch.cat((de_out, en_out), dim=0)
        fu_out = self.trfu(tgt=de_out, memory=fu_patch)





        out = fu_out.squeeze(0).view(b, 500, d).contiguous()


        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(3))
        for i in range(b):
            x[i, inds_copy[i], :] = out[i,:,:]

        x = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        return x


from torch._six import container_abcs
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(rp(x, n))
    return parse
to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x

class EmbedRefine_vis2(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=32, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(EmbedRefine_vis2, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(1, 1, dim_feedforward,
        #                                         dropout, activation, normalize_before)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, 1))
        self.cascade_num = 4
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        # self.trca = nn.ModuleList()
        # for _ in range(self.cascade_num):
        #     self.trca.append(TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
        #                                            dropout, activation, normalize_before))

        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.transform = T.Resize(144)
        self.proj = PatchEmbed(img_size=144, patch_size=48,
                               in_chans=3, embed_dim=3)
        self.conv = nn.Conv2d(3, 1, kernel_size=3)

    def to_patchq(self, x, ind, mask):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # out: [hw,B,d]->[1,66,128]

        x = _tranpose_and_gather_feat(x, ind)  # [3,500,128]
        x = x[mask > 0].contiguous()  # [66,128]
        q_patch = x.unsqueeze(0)
        return q_patch

    def to_patchk(self, x, ind_3x3, mask):
        # x: id feature [B,128,152,272]
        # ind_3x3: center [B,500,9]
        # out: [hw,B,d]->[9,66,128]

        x = _tranpose_and_gather_feat_avgregion(x, ind_3x3, x.size(-1))  # [3,500,128,9]
        x = x[mask > 0].contiguous()  # [66,128,9]
        k_patch = x.permute(2, 0, 1)
        return k_patch

    def forward(self, x, hm=None, wh=None, reg=None, ind=None, ind_3x3=None, mask=None, img=None):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # ind_3x3: center_region [B,500,9]
        # mask: reg_mask [B,500]

        # B = len(ind.nonzero())

        # if ind is not None:
        #     q_patch = self.to_patchq(x, ind, mask)    # [1,66,128]
        #     k_patch = self.to_patchk(x, ind_3x3, mask)   # [9,66,128]
        # else:

        b, d, h, w = x.shape
        device = x.device
        en_patch = torch.randn((b, 500, 1, 1)).to(device)

        dets, inds = mot_decode(hm, wh, reg=reg)
        inds_copy = copy.deepcopy(inds)
        w = x.size(-1)
        inds0 = torch.clamp(inds - w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds1 = torch.clamp(inds - w, 0, 272 * 152 - 1).unsqueeze(2)
        inds2 = torch.clamp(inds - w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds3 = torch.clamp(inds - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds4 = torch.clamp(inds, 0, 272 * 152 - 1).unsqueeze(2)
        inds5 = torch.clamp(inds + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds6 = torch.clamp(inds + w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds7 = torch.clamp(inds + w, 0, 272 * 152 - 1).unsqueeze(2)
        inds8 = torch.clamp(inds + w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds = torch.cat((inds0, inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), dim=2)

        k_patch = _tranpose_and_gather_feat_avgregion(x, inds, w)  # [1,100,128,9]
        # k_patch = k_patch.view(-1, d, 9).contiguous()  # [1500,128,9]

        detections = dets[:, :, :4].int()
        b, n, _ = detections.shape

        for i in range(b):
            for j in range(n):
                tmp = detections[i, j]
                tmp[0] = torch.clamp(tmp[0], 0, 271)
                tmp[1] = torch.clamp(tmp[1], 0, 151)
                tmp[2] = torch.clamp(tmp[2], 0, 271)
                tmp[3] = torch.clamp(tmp[3], 0, 151)
                tw = tmp[2] - tmp[0]
                th = tmp[3] - tmp[1]
                tmp_img = tmp * 4
                if tw <= 0 or th <= 0:
                    continue
                # box = x[i, :, tmp[1]:tmp[3], tmp[0]:tmp[2]]
                box_img = img[i, :, tmp_img[1]:tmp_img[3], tmp_img[0]:tmp_img[2]]
                box_img = F.interpolate(box_img.unsqueeze(0), (144,144))
                tmp_proj = self.proj(box_img)
                en_patch[i, j] = self.conv(tmp_proj).squeeze(0)

                # k_patch[i, j] = self.pool(box).unsqueeze(0).view(128, -1).contiguous()
                # k_patch[i, j] = self.pool(box).view(128, -1).contiguous()
                # print(a.shape)

        k_patch = k_patch.view(-1, d, 9).permute(2, 0, 1).contiguous()
        en_patch = en_patch.view(-1, 1, 1).permute(1, 0, 2).contiguous()
        en_patch = en_patch.squeeze(0).view(b, 500, 1).contiguous()

        q_patch = _tranpose_and_gather_feat(x, inds_copy)  # [3,500,128]
        q_patch = q_patch.view(-1, d).unsqueeze(0) # [1,1500,128]

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b*n).permute(1, 0, 2).contiguous()
        # encoder_patch = torch.cat((cls_tokens, en_patch), dim=0)

        # en_out = self.tr_encoder(src=encoder_patch)[0]

        # q_patch = q_patch + en_out.unsqueeze(0)
        # en_out = en_out.view(b, 500, 1).contiguous()

        out = self.tr_decoder(tgt=q_patch, memory=k_patch)
        out = out.squeeze(0).view(b, 500, d).contiguous()

        # for i in range(self.cascade_num):
        #     out = self.trca[i](tgt=q_patch, memory=k_patch)
        #     q_patch = out
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(3))
        vis = torch.zeros((b, h*w, 1)).to(device)

        for i in range(b):
            x[i, inds_copy[i], :] = out[i,:,:]
            vis[i, inds_copy[i], :] = en_patch[i, :, :]

        x = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        vis = vis.view(b, h, w, 1).permute(0, 3, 1, 2).contiguous()
        # out = out.permute(1, 2, 0).reshape(b, d, h, w)
        return x, vis

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class EmbedRefine_vishm(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(EmbedRefine_vishm, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        #
        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.cascade_num = 4
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        # self.trca = nn.ModuleList()
        # for _ in range(self.cascade_num):
        #     self.trca.append(TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
        #                                            dropout, activation, normalize_before))

        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        self.pool = nn.AdaptiveAvgPool2d(3)
        # self.transform = T.Resize(144)
        # self.proj = PatchEmbed(img_size=144, patch_size=48,
        #                        in_chans=3, embed_dim=128)

    def to_patchq(self, x, ind, mask):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # out: [hw,B,d]->[1,66,128]

        x = _tranpose_and_gather_feat(x, ind)  # [3,500,128]
        x = x[mask > 0].contiguous()  # [66,128]
        q_patch = x.unsqueeze(0)
        return q_patch

    def to_patchk(self, x, ind_3x3, mask):
        # x: id feature [B,128,152,272]
        # ind_3x3: center [B,500,9]
        # out: [hw,B,d]->[9,66,128]

        x = _tranpose_and_gather_feat_avgregion(x, ind_3x3, x.size(-1))  # [3,500,128,9]
        x = x[mask > 0].contiguous()  # [66,128,9]
        k_patch = x.permute(2, 0, 1)
        return k_patch

    def forward(self, x, hm=None, wh=None, reg=None, ind=None, ind_3x3=None, mask=None, vis=None):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # ind_3x3: center_region [B,500,9]
        # mask: reg_mask [B,500]

        # B = len(ind.nonzero())

        # if ind is not None:
        #     q_patch = self.to_patchq(x, ind, mask)    # [1,66,128]
        #     k_patch = self.to_patchk(x, ind_3x3, mask)   # [9,66,128]
        # else:

        b, d, h, w = x.shape
        device = x.device
        en_patch = torch.zeros((b, 500, d)).to(device)

        # vis = _sigmoid(vis)

        dets, inds = mot_decode(hm, wh, reg=reg)
        inds_copy = copy.deepcopy(inds)
        w = x.size(-1)
        inds0 = torch.clamp(inds - w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds1 = torch.clamp(inds - w, 0, 272 * 152 - 1).unsqueeze(2)
        inds2 = torch.clamp(inds - w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds3 = torch.clamp(inds - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds4 = torch.clamp(inds, 0, 272 * 152 - 1).unsqueeze(2)
        inds5 = torch.clamp(inds + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds6 = torch.clamp(inds + w - 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds7 = torch.clamp(inds + w, 0, 272 * 152 - 1).unsqueeze(2)
        inds8 = torch.clamp(inds + w + 1, 0, 272 * 152 - 1).unsqueeze(2)
        inds = torch.cat((inds0, inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), dim=2)

        k_patch = _tranpose_and_gather_feat_avgregion(x, inds, w)  # [1,100,128,9]
        # k_patch = k_patch.view(-1, d, 9).contiguous()  # [1500,128,9]

        detections = dets[:, :, :4].int()
        b, n, _ = detections.shape

        vis_pos = self.pose_encoding(x)
        vis = _tranpose_and_gather_feat(vis, inds_copy)
        vis = (vis * 10).long().view(-1, 1).contiguous()
        vis = vis.unsqueeze(2).expand(vis.size(0), vis.size(1), d)
        vis_pos = vis_pos.unsqueeze(0).expand(b*n, vis_pos.size(0), vis_pos.size(1))
        vis_pos = vis_pos.gather(1, vis).permute(1, 0, 2)


        for i in range(b):
            for j in range(n):
                tmp = detections[i, j]
                tmp[0] = torch.clamp(tmp[0], 0, 271)
                tmp[1] = torch.clamp(tmp[1], 0, 151)
                tmp[2] = torch.clamp(tmp[2], 0, 271)
                tmp[3] = torch.clamp(tmp[3], 0, 151)
                tw = tmp[2] - tmp[0]
                th = tmp[3] - tmp[1]
                # tmp_img = tmp * 4
                if tw <= 0 or th <= 0:
                    continue
                box = x[i, :, tmp[1]:tmp[3], tmp[0]:tmp[2]]
                # box_vis = vis[i, :, tmp[1]:tmp[3], tmp[0]:tmp[2]]
                # box_img = img[i, :, tmp_img[1]:tmp_img[3], tmp_img[0]:tmp_img[2]]
                # box_img = F.interpolate(box_img.unsqueeze(0), (144,144))
                # en_patch[i, j] = self.proj(box_img)

                # k_patch[i, j] = self.pool(box).unsqueeze(0).view(128, -1).contiguous()
                k_patch[i, j] = self.pool(box).view(128, -1).contiguous()
                # print(a.shape)

        k_patch = k_patch.view(-1, d, 9).permute(2, 0, 1).contiguous()
        # en_patch = en_patch.view(-1, d, 9).permute(2, 0, 1).contiguous()

        q_patch = _tranpose_and_gather_feat(x, inds_copy)  # [3,500,128]
        q_patch = q_patch.view(-1, d).unsqueeze(0) # [1,1500,128]

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b*n).permute(1, 0, 2).contiguous()
        # encoder_patch = torch.cat((cls_tokens, en_patch), dim=0)

        # en_out = self.tr_encoder(src=encoder_patch)[0]

        q_patch = q_patch + vis_pos
        # en_out = en_out.view(b, 500, d).contiguous()

        out = self.tr_decoder(tgt=q_patch, memory=k_patch)
        out = out.squeeze(0).view(b, 500, d).contiguous()

        # for i in range(self.cascade_num):
        #     out = self.trca[i](tgt=q_patch, memory=k_patch)
        #     q_patch = out
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(3))
        vis = torch.zeros_like(x)

        for i in range(b):
            x[i, inds_copy[i], :] = out[i,:,:]
            # vis[i, inds_copy[i], :] = en_out[i, :, :]

        x = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        # vis = vis.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        # out = out.permute(1, 2, 0).reshape(b, d, h, w)
        return x

class EmbedRefine_vishm2(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation='relu', normalize_before=False, pos_type='sin'):
        super(EmbedRefine_vishm2, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        #
        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.cascade_num = 4
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        # self.trca = nn.ModuleList()
        # for _ in range(self.cascade_num):
        #     self.trca.append(TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
        #                                            dropout, activation, normalize_before))

        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        self.pool = nn.AdaptiveAvgPool2d(3)
        # self.transform = T.Resize(144)
        # self.proj = PatchEmbed(img_size=144, patch_size=48,
        #                        in_chans=3, embed_dim=128)

    def to_patchq(self, x, ind, mask):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # out: [hw,B,d]->[1,66,128]

        x = _tranpose_and_gather_feat(x, ind)  # [3,500,128]
        x = x[mask > 0].contiguous()  # [66,128]
        q_patch = x.unsqueeze(0)
        return q_patch

    def to_patchk(self, x, ind_3x3, mask):
        # x: id feature [B,128,152,272]
        # ind_3x3: center [B,500,9]
        # out: [hw,B,d]->[9,66,128]

        x = _tranpose_and_gather_feat_avgregion(x, ind_3x3, x.size(-1))  # [3,500,128,9]
        x = x[mask > 0].contiguous()  # [66,128,9]
        k_patch = x.permute(2, 0, 1)
        return k_patch

    def forward(self, x, hm=None, wh=None, reg=None, ind=None, ind_3x3=None, mask=None, vis=None):
        # x: id feature [B,128,152,272]
        # ind: center [B,500]
        # ind_3x3: center_region [B,500,9]
        # mask: reg_mask [B,500]

        # B = len(ind.nonzero())

        # if ind is not None:
        #     q_patch = self.to_patchq(x, ind, mask)    # [1,66,128]
        #     k_patch = self.to_patchk(x, ind_3x3, mask)   # [9,66,128]
        # else:
        vis_pos = self.pose_encoding(x)
        b, d, h, w = x.shape
        device = x.device
        en_patch = torch.zeros((b, 500, d)).to(device)

        # vis = _sigmoid(vis)

        dets, inds = mot_decode(hm, wh, reg=reg)

        detections = dets[:, :, :4].int()
        b, n, _ = detections.shape

        # start = time.time()
        q_all = None
        k_all = None
        num=[0]
        index_all=[]
        for i in range(b):
            d_tmp = detections[i]
            i_tmp = inds[i]

            d_tmp[:, 0] = torch.clamp(d_tmp[:, 0], 0, 271)
            d_tmp[:, 1] = torch.clamp(d_tmp[:, 1], 0, 151)
            d_tmp[:, 2] = torch.clamp(d_tmp[:, 2], 0, 271)
            d_tmp[:, 3] = torch.clamp(d_tmp[:, 3], 0, 151)

            det_tmp = d_tmp[(d_tmp[:, 2] - d_tmp[:, 0]) > 0]
            inds_tmp = i_tmp[(d_tmp[:, 2] - d_tmp[:, 0]) > 0]
            d_tmp = det_tmp[(det_tmp[:, 3] - det_tmp[:, 1]) > 0]
            i_tmp = inds_tmp[(det_tmp[:, 3] - det_tmp[:, 1]) > 0]
            num.append(len(d_tmp))
            index_all.append(i_tmp)
            if len(d_tmp) == 0:
                continue
            ############################## 0.005

            list = [x[i, :, d_tmp[t, 1]:d_tmp[t, 3], d_tmp[t, 0]:d_tmp[t, 2]] for t in range(len(d_tmp))]
            k_list = [self.pool(list[t]).view(128, -1).permute(1, 0).unsqueeze(1) for t in range(len(list))]
            k_tmp = k_list[0]
            for j in range(len(k_list)-1):
                k_tmp = torch.cat((k_tmp, k_list[j+1]), dim=1)

            q_tmp = _tranpose_and_gather_feat(x[i].unsqueeze(0), i_tmp.unsqueeze(0))

            vis_tmp = _tranpose_and_gather_feat(vis[i].unsqueeze(0), i_tmp.unsqueeze(0))
            vis_tmp = (vis_tmp * 10).long().view(-1, 1).contiguous()
            vis_tmp = vis_tmp.unsqueeze(2).expand(vis_tmp.size(0), vis_tmp.size(1), d)
            vis_pos_tmp = vis_pos.unsqueeze(0).expand(vis_tmp.size(0), vis_pos.size(0), vis_pos.size(1))
            vis_pos_tmp = vis_pos_tmp.gather(1, vis_tmp).permute(1, 0, 2)

            q_tmp = q_tmp + vis_pos_tmp

            if q_all is None:
                q_all = q_tmp
                k_all = k_tmp
            else:
                q_all = torch.cat((q_all, q_tmp), dim=1)
                k_all = torch.cat((k_all, k_tmp), dim=1)

        if q_all is not None:
            out = self.tr_decoder(tgt=q_all, memory=k_all).squeeze(0)
            out_list = [out[num[i]: (num[i] + num[i + 1]), :] for i in range(b)]



            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, x.size(3))


            # index_re = torch.arange(h * w).to(device)
            # re_list = torch.Tensor([l for l in index_re if l not in index_all[0]]).long().to(device)
            #
            # new = torch.zeros((b, h * w, 128)).to(device)
            #
            # new[0, re_list, :] = x[0, re_list, :]
            # new[0, index_all[0], :] = out_list[0]


            for i in range(b):
                x[i, index_all[i], :] = out_list[i]


            # end =time.time()
            # print(end-start)


        x = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        # new = new.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        return x

class TransFusion4(nn.Module):  # concat
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False,
                 norm_cfg=dict(type='BN', requires_grad=True), overlap=False, pos_type='sin'):
        super(TransFusion4, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        if overlap:
            stride_size1 = (patch_size1[0] // 2, patch_size1[1] // 2)
            stride_size2 = (patch_size2[0] // 2, patch_size2[1] // 2)
        else:
            stride_size1 = patch_size1
            stride_size2 = patch_size2
        self.to_patch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=patch_size1, stride=stride_size1),
            nn.Conv2d(in_chans1, d_model, kernel_size=1, stride=1),
            # nn.BatchNorm2d(d_model),
        )
        # self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=patch_size2, stride=stride_size2),
            nn.Conv2d(in_chans2, d_model, kernel_size=1, stride=1),
            # nn.BatchNorm2d(d_model),
        )
        self.out_conv = ConvModule(
            d_model + in_chans1,
            in_chans2,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2):
        # x1: input list
        # x2: output
        # patchs = []
        # pos1 = []
        # for i in range(len(x1)):
        #     patchs.append(self.to_patch1[i](x1[i]))
        k_patch = self.to_patch1(x1)
        q_patch = self.to_patch2(x2)
        b, d, h, w = q_patch.shape
        # for patch in patchs:
        #     pos1.append(self.pose_encoding(patch).transpose(0,1))
        pos1 = self.pose_encoding(k_patch).transpose(0, 1)    # hw,B,d
        # pos1 = torch.zeros(h*w, b, d, dtype=tmp.dtype, device=tmp.device)
        # patchs = torch.cat([patch.flatten(2).permute(2,0,1) for patch in patchs], dim=0)
        # pos1 = torch.cat([p for p in pos1], dim=0)
        pos2 = self.pose_encoding(q_patch).transpose(0, 1)     # hw,B,d
        # pos2 = torch.zeros(h*w, b, d, dtype=patchs.dtype, device=patchs.device)
        # src = x1.flatten(2).permute(2,0,1)
        # memory = self.tr_encoder(src=src, pos=pos1)
        k_patch = k_patch.flatten(2).permute(2, 0, 1)    # hw,B,d
        tgt = q_patch.flatten(2).permute(2, 0, 1)      # hw,B,d
        # out = self.tr_decoder(tgt=tgt, memory=src, pos=pos1, query_pos=pos2)
        out = self.tr_decoder(tgt=tgt, memory=k_patch, pos=pos1, query_pos=pos2)
        # out = self.tr_decoder(tgt=patchs, memory=tgt, pos=pos2, query_pos=pos1)
        out = out.permute(1, 2, 0).reshape(b, d, h, w)
        # out = self.out_conv(torch.cat(
        #     (x2, F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1))
        # out = F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + x2
        out = self.out_conv(
            torch.cat((F.interpolate(x1, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True),
                       F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)),
                      dim=1)
        )
        return out

class TransFusion5(nn.Module):  # add
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False,
                 norm_cfg=dict(type='BN', requires_grad=True), overlap=False, pos_type='sin', adapt_size1=(10, 10),
                 adapt_size2=(10, 10)):
        super(TransFusion5, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn()  ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()  ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        if overlap:
            stride_size1 = (patch_size1[0] // 2, patch_size1[1] // 2)
            stride_size2 = (patch_size2[0] // 2, patch_size2[1] // 2)
        else:
            stride_size1 = patch_size1
            stride_size2 = patch_size2
        self.to_patch1 = nn.Sequential(
            # nn.AvgPool2d(kernel_size=patch_size1, stride=stride_size1),
            nn.AdaptiveAvgPool2d(adapt_size1),
            nn.Conv2d(in_chans1, d_model, kernel_size=1, stride=1),
            # nn.BatchNorm2d(d_model),
        )
        # self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Sequential(
            # nn.AvgPool2d(kernel_size=patch_size2, stride=stride_size2),
            nn.AdaptiveAvgPool2d(adapt_size2),
            nn.Conv2d(in_chans2, d_model, kernel_size=1, stride=1),
            # nn.BatchNorm2d(d_model),
        )
        self.out_conv = ConvModule(
            d_model,
            in_chans2,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.out_conv2 = ConvModule(
            in_chans1,
            in_chans2,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2):
        # x1: input list
        # x2: output
        # patchs = []
        # pos1 = []
        # for i in range(len(x1)):
        #     patchs.append(self.to_patch1[i](x1[i]))
        k_patch = self.to_patch1(x1)
        q_patch = self.to_patch2(x2)
        b, d, h, w = q_patch.shape
        # for patch in patchs:
        #     pos1.append(self.pose_encoding(patch).transpose(0,1))
        pos1 = self.pose_encoding(k_patch).transpose(0, 1)
        # pos1 = torch.zeros(h*w, b, d, dtype=tmp.dtype, device=tmp.device)
        # patchs = torch.cat([patch.flatten(2).permute(2,0,1) for patch in patchs], dim=0)
        # pos1 = torch.cat([p for p in pos1], dim=0)
        pos2 = self.pose_encoding(q_patch).transpose(0, 1)
        # pos2 = torch.zeros(h*w, b, d, dtype=patchs.dtype, device=patchs.device)
        # src = x1.flatten(2).permute(2,0,1)
        # memory = self.tr_encoder(src=src, pos=pos1)
        k_patch = k_patch.flatten(2).permute(2, 0, 1)
        tgt = q_patch.flatten(2).permute(2, 0, 1)
        # out = self.tr_decoder(tgt=tgt, memory=src, pos=pos1, query_pos=pos2)
        out = self.tr_decoder(tgt=tgt, memory=k_patch, pos=pos1, query_pos=pos2)
        # out = self.tr_decoder(tgt=patchs, memory=tgt, pos=pos2, query_pos=pos1)
        out = out.permute(1, 2, 0).reshape(b, d, h, w)
        # out = self.out_conv(torch.cat(
        #     (x2, F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1))
        # out = F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + x2
        tmp = F.interpolate(self.out_conv2(x1), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)
        out = F.interpolate(self.out_conv2(x1), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear',
                            align_corners=True) + \
              F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)
        return out