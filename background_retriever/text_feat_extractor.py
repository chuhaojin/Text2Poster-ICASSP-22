# -*- encoding: utf-8 -*-
'''
@File    :   text_feat_extractor.py
@Time    :   2021/08/26 10:46:15
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import os
import sys
base_dir = os.path.abspath(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from utils import getLanMask
from utils.config import cfg_from_yaml_file, cfg
import yaml
from easydict import EasyDict
from models.TextEncoder import TextLearnableEncoder


class TextModel(nn.Module):
    def __init__(self, model_cfg):
        super(TextModel, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable["textencoder"] = TextLearnableEncoder(model_cfg)

    def forward(self, texts, maskTexts):
        textFea = self.learnable["textencoder"](texts, maskTexts) # <bsz, img_dim>
        textFea = F.normalize(textFea, p=2, dim=-1)
        return textFea


class TextFeatureExtractor(nn.Module):
    def __init__(self, cfg_file,):
        super(TextFeatureExtractor, self).__init__()
        self.cfg_file = cfg_file
        cfg = EasyDict()
        self.cfg = cfg_from_yaml_file(self.cfg_file, cfg)
        self.cfg.MODEL.ENCODER = os.path.join(base_dir, self.cfg.MODEL.ENCODER)
        self.text_model = TextModel(model_cfg=self.cfg.MODEL)
        self.text_transform = AutoTokenizer.from_pretrained(self.cfg.MODEL.ENCODER)

    def extract(self, text_input):
        if text_input is None:
            return None
        else:
            text_info = self.text_transform(text_input, padding='max_length', truncation=True,
                                            max_length=self.cfg.MODEL.MAX_TEXT_LEN, return_tensors='pt')
            text = text_info.input_ids.reshape(-1)
            text_len = torch.sum(text_info.attention_mask)
            with torch.no_grad():
                texts = text.unsqueeze(0) 
                text_lens = text_len.unsqueeze(0)
                textMask = getLanMask(text_lens, self.cfg.MODEL.MAX_TEXT_LEN)
                textMask = textMask.cuda()
                texts = texts.cuda()
                text_lens = text_lens.cuda()
                text_fea = self.text_model(texts, textMask)
                text_fea = text_fea.cpu().numpy()
            return text_fea
    
    def forward(self, inputs):
        return inputs

if __name__ == '__main__':

    # The text query.
    text_query = "北京的秋天是真的凉爽。"

    # The config of BriVL model.
    cfg_file = os.path.join(base_dir, 'cfg/BriVL_cfg.yml')
    
    # Text Encoder of BriVL model.
    text_extractor = TextFeatureExtractor(cfg_file)
    model_component = torch.load("weights/brivl-textencoder-weights.pth")
    text_extractor.load_state_dict(model_component)
    text_extractor.eval()
    text_extractor.cuda()
    
    print("text query:", text_query)
    fea = text_extractor.extract(text_query)
    # fea is a 2048-dimensional vector.
    print("fea:", fea)
