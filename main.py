# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/01/05 22:05:59
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import argparse
import json
import cv2
import numpy as np
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# retrieve the background image based on given text.
from background_retrieval import bk_img_retrieval
# Tool for write text on image.
from utils.font_utils import PutText2Image

# the layout distribution predict model.
from model.distrib_model import LayoutsDistribModel
# the layout refinement model.
from model.layout_model import BBoxesRegModel

# smooth region detection and predict the distribution of layout.
from layout_distribution_predict import smooth_region_dectection, get_distrib_mask

# refine the layout bounding boxxes.
from layout_refine import get_batch_text_region, get_bbox_mask, get_refine_bboxes

STD_WIDTH, STD_HEIGHT = 300, 400
MIN_VALUE = -999999
MAX_BBOX_NUM = 32


def draw_figure(mask_list, layers, height, width):
    figure_mask = np.zeros((height, width, 3))
    for mask, layer in zip(mask_list, layers):
        figure_mask[:, :, layer] = mask / mask.max()
    figure_mask = cv2.resize(figure_mask, (width, height))
    return figure_mask * 255


def save_process_to_figure(
    saliency_map, 
    smooth_region_mask, 
    bbox_distrib_map, 
    initial_bbox_mask,
    refined_bbox_mask,
    height, width,
    save_folder):
    '''
    Save the poster generation process to figure, such as sailency map and etc.
    '''
    initial_bbox_mask = cv2.resize(initial_bbox_mask[0], (width, height))
    smooth_region_mask = cv2.resize(smooth_region_mask[0], (width, height))
    bbox_distrib_map = cv2.resize(bbox_distrib_map, (width, height))
    refined_bbox_mask = cv2.resize(refined_bbox_mask[0], (width, height))

    cv2.imwrite(os.path.join(save_folder, "candidate_regions.jpg"), smooth_region_mask * 255)
    cv2.imwrite(os.path.join(save_folder, "layout_distribution.jpg"), bbox_distrib_map * 255)

    smooth_region_figure = draw_figure(
        [saliency_map, smooth_region_mask], 
        [0, 2],
        height, width)
    cv2.imwrite(os.path.join(save_folder, "saliency_map_with-smooth.jpg"), smooth_region_figure)

    layout_distribution_figure = draw_figure(
        [saliency_map, bbox_distrib_map], 
        [0, 2],
        height, width)
    cv2.imwrite(os.path.join(save_folder, "saliency_map_with-layout-distribution.jpg"), layout_distribution_figure)

    initial_layout_figure = draw_figure(
        [initial_bbox_mask, bbox_distrib_map, saliency_map], 
        [2, 1, 0],
        height, width)
    cv2.imwrite(os.path.join(save_folder, "initial_layout.jpg"), initial_layout_figure)

    refined_layout_figure = draw_figure(
        [refined_bbox_mask, bbox_distrib_map, saliency_map], 
        [2, 1, 0],
        height, width)
    cv2.imwrite(os.path.join(save_folder, "refined_layout.jpg"), refined_layout_figure)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text_file", type=str, dest = "input_text_file", default="example/input_text_elements_1.json")
    parser.add_argument("--output_folder", type=str, dest = "output_folder", default="example/outputs_1")
    parser.add_argument("--background_folder", type=str, dest = "background_folder", default="bk_image_folder")
    parser.add_argument("--save_process", action='store_true')
    parser.add_argument("--top_n", type=int, dest = "top_n", default=5)
    args = parser.parse_args()
    print(args)

    # The iteration round of layout refine model.
    iteration_rounds = 30 
    font_file = "./font_files/test_font.TTF"
    text_color = (255, 255 ,255)
    ft_center = PutText2Image(font_file)

    if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)

    # load the input text elements.
    f = open(args.input_text_file, "r")
    input_text_elements = json.load(f) # [(text_1, font_size_n),...,(text_n, font_size_n)]
    f.close()
    print(input_text_elements)
    # sort sentences based on font size.
    sentences = input_text_elements["sentences"]
    sentences = sorted(sentences, reverse = True, key = lambda x: x[1])
    data_len = len(sentences)

    for text_info in sentences:
        print("Text: {}, Font Size:{}".format(text_info[0], text_info[1]))
    
    # retrieve the background image based on given text.
    # We use the text with max font size to retrieve background image.
    background_query = input_text_elements["background_query"]
    print("Background retrieval query:", background_query)
    # Our Text2Poster is built based on BriVL (CLIP in Chinese). 
    # If you need to retrieve background images based on English, you can use CLIP.
    image_path_list = bk_img_retrieval(background_query, args.background_folder)

    for i, img_path in enumerate(tqdm(image_path_list[:args.top_n])):
        print("background image: ", img_path)
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        img_size = (width, height)
        smooth_region_mask, regions, saliency_map = smooth_region_dectection(img)
        bbox_distrib_map = get_distrib_mask(smooth_region_mask)
        bbox_size_array = np.zeros((len(sentences), 2))

        # Estimate the size of the text box.
        for j, text_info in enumerate(sentences):
            bbox_size_array[j] = ft_center.get_text_bbox_size(text=text_info[0], text_size=text_info[1])

        # initial layout, sampled by the maximum probability above the bbox_distrib_map.
        initial_bboxes = get_batch_text_region(bbox_distrib_map, bbox_size_array, img_size)
        initial_bbox_mask = get_bbox_mask(initial_bboxes, data_len)

        # The data used to refine the layout.
        initial_data = {"len_info": data_len, 
                    "shifted_mask": initial_bbox_mask.copy(),
                    "shifted_bbox": initial_bboxes.copy(),
                    "bbox_distrib_map": bbox_distrib_map.copy(),
                    "smooth_region_mask": smooth_region_mask.copy()}
        
        # The refined layout.
        refined_bboxes, refined_bbox_size, order = get_refine_bboxes(initial_data, iteration_rounds)
        refined_bbox_mask = get_bbox_mask(refined_bboxes[None, :], data_len)

        # scale the layout to the image size.
        refined_bboxes[:, (0, 2)] = refined_bboxes[:, (0, 2)] / STD_WIDTH * width
        refined_bboxes[:, (1, 3)] = refined_bboxes[:, (1, 3)] / STD_HEIGHT * height

        # save the final poster picture.
        bk_img = img.copy()
        bk_img = bk_img / 1.1
        bk_img = bk_img.astype(np.uint8)
        for j in range(len(sentences)):
            text_position = (refined_bboxes[j][0], refined_bboxes[j][1])
            text, text_size = sentences[order[j]][0], sentences[order[j]][1]
            bk_img = ft_center.draw_text(bk_img, text_position, text, text_size, text_color)
        save_folder = os.path.join(args.output_folder, str(i))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        poster_file = os.path.join(save_folder, "poster.jpg")
        cv2.imwrite(poster_file, bk_img)

        if args.save_process:
            save_process_to_figure(
                saliency_map, 
                smooth_region_mask, 
                bbox_distrib_map, 
                initial_bbox_mask,
                refined_bbox_mask,
                height, width,
                save_folder)
    return


if __name__ == "__main__":
    main()
