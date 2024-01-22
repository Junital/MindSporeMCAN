# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

#from core.model.net_utils import FC, MLP, LayerNorm
#
#import torch.nn as nn
#import torch.nn.functional as F
#import torch, math




# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

#lass MHAtt(nn.Module):
#   def __init__(self, __C):
#       super(MHAtt, self).__init__()
#       self.__C = __C

#       self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
#       self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
#       self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
#       self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

#       self.dropout = nn.Dropout(__C.DROPOUT_R)

#   def forward(self, v, k, q, mask):
#       n_batches = q.size(0)

#       v = self.linear_v(v).view(
#           n_batches,
#           -1,
#           self.__C.MULTI_HEAD,
#           self.__C.HIDDEN_SIZE_HEAD
#       ).transpose(1, 2)

#       k = self.linear_k(k).view(
#           n_batches,
#           -1,
#           self.__C.MULTI_HEAD,
#           self.__C.HIDDEN_SIZE_HEAD
#       ).transpose(1, 2)

#       q = self.linear_q(q).view(
#           n_batches,
#           -1,
#           self.__C.MULTI_HEAD,
#           self.__C.HIDDEN_SIZE_HEAD
#       ).transpose(1, 2)

#       atted = self.att(v, k, q, mask)
#       atted = atted.transpose(1, 2).contiguous().view(
#           n_batches,
#           -1,
#           self.__C.HIDDEN_SIZE
#       )

#       atted = self.linear_merge(atted)

#       return atted

#   def att(self, value, key, query, mask):
#       d_k = query.size(-1)

#       scores = torch.matmul(
#           query, key.transpose(-2, -1)
#       ) / math.sqrt(d_k)

#       if mask is not None:
#           scores = scores.masked_fill(mask, -1e9)

#       att_map = F.softmax(scores, dim=-1)
#       att_map = self.dropout(att_map)

#       return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

#lass FFN(nn.Module):
#   def __init__(self, __C):
#       super(FFN, self).__init__()

#       self.mlp = MLP(
#           in_size=__C.HIDDEN_SIZE,
#           mid_size=__C.FF_SIZE,
#           out_size=__C.HIDDEN_SIZE,
#           dropout_r=__C.DROPOUT_R,
#           use_relu=True
#       )

#   def forward(self, x):
#       return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

#lass SA(nn.Module):
#   def __init__(self, __C):
#       super(SA, self).__init__()

#       self.mhatt = MHAtt(__C)
#       self.ffn = FFN(__C)

#       self.dropout1 = nn.Dropout(__C.DROPOUT_R)
#       self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

#       self.dropout2 = nn.Dropout(__C.DROPOUT_R)
#       self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

#   def forward(self, x, x_mask):
#       x = self.norm1(x + self.dropout1(
#           self.mhatt(x, x, x, x_mask)
#       ))

#       x = self.norm2(x + self.dropout2(
#           self.ffn(x)
#       ))

#       return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

#lass SGA(nn.Module):
#   def __init__(self, __C):
#       super(SGA, self).__init__()

#       self.mhatt1 = MHAtt(__C)
#       self.mhatt2 = MHAtt(__C)
#       self.ffn = FFN(__C)

#       self.dropout1 = nn.Dropout(__C.DROPOUT_R)
#       self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

#       self.dropout2 = nn.Dropout(__C.DROPOUT_R)
#       self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

#       self.dropout3 = nn.Dropout(__C.DROPOUT_R)
#       self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

#   def forward(self, x, y, x_mask, y_mask):
#       x = self.norm1(x + self.dropout1(
#           self.mhatt1(x, x, x, x_mask)
#       ))

#       x = self.norm2(x + self.dropout2(
#           self.mhatt2(y, y, x, y_mask)
#       ))

#       x = self.norm3(x + self.dropout3(
#           self.ffn(x)
#       ))

#       return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

#lass MCA_ED(nn.Module):
#   def __init__(self, __C):
#       super(MCA_ED, self).__init__()

#       self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
#       self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

#   def forward(self, x, y, x_mask, y_mask):
#       # Get hidden vector
#       for enc in self.enc_list:
#           x = enc(x, x_mask)

#       for dec in self.dec_list:
#           y = dec(y, x, y_mask, x_mask)

#       return x, y

import mindspore
from mindspore import nn,Tensor,Parameter,ops
import numpy as np
import os

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
class MHAtt(nn.Cell):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Dense(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Dense(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Dense(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Dense(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(p=__C.DROPOUT_R)
    def att(self, value, key, query, mask=None):
        d_k = query.shape[-1]
        if len(key.shape)==3:
            k_copy=ops.transpose(key,(0,1,2))
        else:
            k_copy=ops.transpose(key,(0,1,3,2))
                                 
        scores = ops.matmul(
            query, k_copy
        ) / ops.Sqrt()(Tensor(d_k,mindspore.float32))

        if mask is not None:
            scores = ops.masked_fill(scores,mask, -1e9)

        att_map = ops.softmax(scores)
        att_map = self.dropout(att_map)

        return ops.matmul(att_map, value)
    
    
    def construct(self,v,k,q,mask=None):
        n_batches = q.shape[0]

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        )
        v=ops.transpose(v,(0,2,1,3))

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        )
        k=ops.transpose(k,(0,2,1,3))

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        )
        q=ops.transpose(q,(0,2,1,3))

        atted = self.att(v, k, q, mask)
        atted=ops.transpose(atted,(0,2,1,3))
        #atted.contiguous()
        atted=atted.view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted




# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Cell):
   def __init__(self, __C):
       super(FFN, self).__init__()

       self.mlp = MLP(
           in_size=__C.HIDDEN_SIZE,
           mid_size=__C.FF_SIZE,
           out_size=__C.HIDDEN_SIZE,
           dropout_r=__C.DROPOUT_R,
           use_relu=True
       )

   def construct(self, x):
       return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Cell):
   def __init__(self, __C):
       super(SA, self).__init__()

       self.mhatt = MHAtt(__C)
       self.ffn = FFN(__C)

       self.dropout1 = nn.Dropout(p=__C.DROPOUT_R)
       self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

       self.dropout2 = nn.Dropout(p=__C.DROPOUT_R)
       self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

   def construct(self, x, x_mask):
       x = self.norm1(x + self.dropout1(
           self.mhatt(x, x, x, x_mask)
       ))
       x = self.norm2(x + self.dropout2(
           self.ffn(x)
       ))

       return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Cell):
   def __init__(self, __C):
       super(SGA, self).__init__()

       self.mhatt1 = MHAtt(__C)
       self.mhatt2 = MHAtt(__C)
       self.ffn = FFN(__C)

       self.dropout1 = nn.Dropout(p=__C.DROPOUT_R)
       self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

       self.dropout2 = nn.Dropout(p=__C.DROPOUT_R)
       self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

       self.dropout3 = nn.Dropout(p=__C.DROPOUT_R)
       self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

   def construct(self, x, y, x_mask, y_mask):
       x = self.norm1(x + self.dropout1(
           self.mhatt1(x, x, x, x_mask)
       ))

       x = self.norm2(x + self.dropout2(
           self.mhatt2(y, y, x, y_mask)
       ))

       x = self.norm3(x + self.dropout3(
           self.ffn(x)
       ))

       return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Cell):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.CellList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.CellList([SGA(__C) for _ in range(__C.LAYER)])

    def construct(self, x, y, x_mask, y_mask):
       # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y