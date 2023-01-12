# -*- encoding: utf-8 -*-
'''
@File    :   vl_model.py
@Time    :   2023/01/12 12:20:28
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import torch
import torch.nn as nn
from .fakeTransformer import FakeTransformer
from .bert import Bert
import torch.nn.functional as F
import timm
import numpy as np
import sys

class TextLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TextLearnableEncoder, self).__init__()

        self.backbone = Bert(model_cfg)
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['textFC'] = FakeTransformer(model_cfg.TEXT_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM)
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.TEXT_FEATURE_DIM, nhead=model_cfg.TEXT_TRANSFORMER_HEAD)
        self.learnable['textAtt'] = nn.TransformerEncoder(text_encoder_layer, num_layers=model_cfg.TEXT_TRANSFORMER_LAYER)

        self.init_param()

    def init_param(self):
        for name, param in self.backbone.named_parameters():
            if 'large' not in self.model_cfg.ENCODER:

                if 'layer.11' not in name and 'layer.10' not in name and 'layer.9' not in name and 'layer.8' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                if 'layer.21' not in name and 'layer.22' not in name and 'layer.23' not in name and 'layer.20' not in name: #  and 'layer.9' not in name
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        sys.stdout.flush()
        

    def forward(self, textFea, maskTexts):

        textFea = self.backbone(textFea)

        textFea = F.normalize(textFea, p=2, dim=-1)
        textFea = self.learnable['textAtt'](textFea.transpose(0, 1), src_key_padding_mask=(maskTexts == 0)).transpose(0,1)
        tmpMask = torch.where(maskTexts == 1, torch.tensor([1.0], device=maskTexts.device),
                              torch.tensor([0.0], device=maskTexts.device))
        textFea = (textFea * tmpMask.unsqueeze(-1)).sum(dim=1) / tmpMask.sum(dim=1).unsqueeze(-1)
        textFea = self.learnable['textFC'](textFea)
        return textFea

