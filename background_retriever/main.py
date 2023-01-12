# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/01/12 13:00:36
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import numpy as np
import jsonlines
import os
import torch
from text_feat_extractor import TextFeatureExtractor


def load_bk_data(bk_fea_file, bk_url_file):
    bk_url_list = []
    with jsonlines.open(bk_url_file, mode = "r") as reader:
        for obj in reader:
            bk_url_list.append(obj)
    bk_feats = np.load(bk_fea_file)
    return bk_feats, bk_url_list


def main():
    # The text query.
    text_query = "北京的秋天是真的凉爽。"

    # The config of BriVL model.
    cfg_file = './cfg/BriVL_cfg.yml'


    # Load the feature and url of background images.
    bk_base_folder = "./background_feats"
    bk_fea_file = os.path.join(bk_base_folder, "wenlan_unsplash_feats.npy")
    bk_url_file = os.path.join(bk_base_folder, "unsplash_image_url.jsonl")
    bk_feats, bk_url_list = load_bk_data(bk_fea_file, bk_url_file)

    # Text Encoder of BriVL model.
    text_extractor = TextFeatureExtractor(cfg_file)
    model_component = torch.load("./weights/brivl-textencoder-weights.pth")
    text_extractor.load_state_dict(model_component)
    text_extractor.eval()
    text_extractor.cuda()
    
    print("text query:", text_query)
    query_feat = text_extractor.extract(text_query)

    cosine_scores = (query_feat * bk_feats).sum(axis = 1)

    top_10_index = (-cosine_scores).argsort()[:10]
    for index in top_10_index:
        image_url = bk_url_list[index]
        print(image_url)
    return

if __name__ == "__main__":
    main()
