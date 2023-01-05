# -*- encoding: utf-8 -*-
'''
@File    :   layout_refine.py
@Time    :   2023/01/05 16:18:00
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from utils.anchor_utils import get_text_region
from model.layout_model import BBoxesRegModel

STD_WIDTH, STD_HEIGHT = 300, 400
channel_deep = 64
MIN_VALUE = -999999
MAX_BBOX_NUM = 32

layout_model = BBoxesRegModel(channel_deep = channel_deep)
ckpt_path = "./checkpoint/0.20484_Cascading_128_uniform_big.pth"
layout_model.load_state_dict(torch.load(ckpt_path).module.state_dict())
layout_model.cuda()
layout_model.eval()


# Given the bounding box of a layout, return the layout binary mask.
def get_bbox_mask(bbox, data_len):
    mask = np.zeros((1, STD_HEIGHT, STD_WIDTH), dtype = np.uint8)
    regions = np.array([[[obj[0], obj[1]], 
                        [obj[2], obj[1]], 
                        [obj[2], obj[3]],
                        [obj[0], obj[3]]] for obj in bbox[0][:min(MAX_BBOX_NUM, data_len)]], dtype = np.int32)
    cv2.fillPoly(mask[0], regions, 1, 1)
    return mask


# 
def get_batch_text_region(distrib_mask, bbox_size_list, img_size):
    temp_mask = distrib_mask.copy().astype(np.float64)
    bbox_pos = np.zeros((1, MAX_BBOX_NUM, 4))
    pixel_gap = STD_HEIGHT // 10
    for i, bbox_size in enumerate(bbox_size_list):
        scaled_text_width = int(bbox_size[0] / img_size[0] * STD_WIDTH)
        scaled_text_height = int(bbox_size[1] / img_size[1] * STD_HEIGHT)
        text_bboxes, text_scores = get_text_region(temp_mask, (scaled_text_width, scaled_text_height), top_n=1)

        text_regions = np.array([[[obj[0] - pixel_gap, obj[1] - pixel_gap], 
                                  [obj[2]+pixel_gap, obj[1] - pixel_gap], 
                                  [obj[2]+pixel_gap, obj[3]+pixel_gap], 
                                  [obj[0] - pixel_gap, obj[3]+pixel_gap]] 
                         for obj in text_bboxes])
    
        cv2.fillPoly(temp_mask, text_regions, -10)
        bbox_pos[0, i] = text_bboxes[0]
    return bbox_pos


# used to deoverlap.
def overlap_detection(bboxes, box_id = -1):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = bboxes.size(0)
    B = bboxes.size(0)
    max_xy = torch.min(bboxes[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bboxes[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(bboxes[:, :2].unsqueeze(1).expand(A, B, 2),
                       bboxes[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    res = inter[:, :, 0] * inter[:, :, 1]
    diag = torch.diag(res)
    res_diag = torch.diag_embed(diag)
    res = res - res_diag
    if box_id == -1:
        if res.max() != 0:
            return True
        else:
            return False
    if res[box_id].max() != 0:
        overlap_id = torch.argmax(res[box_id])
        return True, overlap_id
    else:
        return False, -1


# Given the current layout, predict the refine layout.
def get_next_bbox(cur_bbox, shift, data_len, x_step, y_step, bbox_size, order):

    move_pixel = 10
#     shift[np.abs(shift)<0.1] = 0
    if np.abs(shift).sum() == 0:
        return cur_bbox, True
    cur_bbox = cur_bbox * np.array((STD_WIDTH, STD_HEIGHT, STD_WIDTH, STD_HEIGHT))
    cur_bbox = cur_bbox.astype(np.int32)
    
    shift_scaled = shift *  np.array((x_step, y_step))
    next_bbox = np.zeros((1, MAX_BBOX_NUM, 4), dtype = np.int32)
    next_bbox[0, :data_len] = cur_bbox[0, :data_len]
#     print(bbox_size)
    for i in range(data_len):
        next_bbox[0, i, :2] = cur_bbox[0, i, :2] + shift_scaled[0,i]

        next_bbox[0, i, 2:] = next_bbox[0, i, :2] + bbox_size[i]
        isoverlap, overlap_id = overlap_detection(torch.tensor(next_bbox[0]), box_id = i)
        if isoverlap:
            next_bbox[0, i, :2] = cur_bbox[0, i, :2]
            next_bbox[0, i, 0] += shift_scaled[0, i, 0]
            next_bbox[0, i, 0] = max(min(STD_WIDTH - bbox_size[i, 0], next_bbox[0, i, 0]), 0)
            if next_bbox[0, i, 1] < next_bbox[0, overlap_id, 1]:
                if next_bbox[0, i, 1] > move_pixel:
                    next_bbox[0, i, 1] -= move_pixel
                else:
                    next_bbox[0, overlap_id, 1] += move_pixel
                    next_bbox[0, overlap_id, 3] += move_pixel
            elif next_bbox[0, i, 1] >= next_bbox[0, overlap_id, 1]:
                if next_bbox[0, i, 1] <=STD_HEIGHT - move_pixel - bbox_size[i, 0]:
                    next_bbox[0, i, 1] += move_pixel
                else:
                    next_bbox[0, overlap_id, 1] -= move_pixel
                    next_bbox[0, overlap_id, 3] -= move_pixel

            next_bbox[0, i, 2:] = next_bbox[0, i, :2] + bbox_size[i]

    next_bbox[0, data_len:] = MIN_VALUE
    bbox_argsort = np.argsort(-next_bbox[0, :, 1])
    next_bbox[0] = next_bbox[0, bbox_argsort]
    bbox_size = bbox_size[bbox_argsort]
    order = order[bbox_argsort]
    next_bbox[0, data_len:] = 0
    return next_bbox, bbox_size, order


# refine the layout iteratively.
def get_refine_bboxes(initial_data, iteration_rounds = 10):
    len_info, bbox_mask, shifted_bbox = initial_data["len_info"], initial_data["shifted_mask"], initial_data["shifted_bbox"]
    bbox_distrib_map, smooth_region_mask = initial_data["bbox_distrib_map"], initial_data["smooth_region_mask"]
    data_len = min(len_info, len_info)
    shifted_bbox[0, len_info: ] = MIN_VALUE
    sort_order = np.argsort(-shifted_bbox[0, :, 1])
    shifted_bbox[0, :, ] = shifted_bbox[0, sort_order]
    shifted_bbox[0, len_info: ] = 0
    init_bbox_size = shifted_bbox[0, :, 2:] - shifted_bbox[0, :, :2]
    len_info = torch.tensor(len_info).unsqueeze(0)
    shifted_bbox = torch.tensor(shifted_bbox) / torch.tensor((STD_WIDTH, STD_HEIGHT, STD_WIDTH, STD_HEIGHT))
    shifted_bbox = shifted_bbox.float().cuda()
    
    bbox_distrib_map = torch.tensor(bbox_distrib_map).float().unsqueeze(0).unsqueeze(0).cuda()
    bbox_mask = torch.tensor(bbox_mask).float().cuda()
    smooth_region_mask = torch.tensor(smooth_region_mask).float().cuda()
    len_info = len_info.cuda()
    order = np.zeros((MAX_BBOX_NUM, ), dtype = np.int32)
    order[:len_info] = sort_order[:len_info]

    for _ in range(iteration_rounds):
        with torch.no_grad():
            shifted_pred = layout_model.forward(len_info, bbox_mask, shifted_bbox, None,
                                             bbox_distrib_map, smooth_region_mask, None, inference = True)
        next_bbox, init_bbox_size, order = get_next_bbox(shifted_bbox.clone().cpu().numpy(), 
                                                   shifted_pred.clone().cpu().numpy(), 
                                                   data_len, x_step = STD_WIDTH // 10, y_step = STD_HEIGHT // 10, 
                                                   bbox_size = init_bbox_size, order = order)
        next_mask = get_bbox_mask(next_bbox, data_len)
        shifted_bbox = torch.tensor(next_bbox) / torch.tensor((STD_WIDTH, STD_HEIGHT, STD_WIDTH, STD_HEIGHT))
        shifted_bbox = shifted_bbox.cuda()
        shifted_mask = torch.tensor(next_mask).cuda()
    return next_bbox[0], init_bbox_size[0], order


if __name__ == "__main__":
    from utils.font_utils import PutText2Image
    from model.distrib_model import LayoutsDistribModel
    from layout_distribution_predict import smooth_region_dectection, get_distrib_mask

    # A example of layout text.
    sentences = [
    ("Hello, World", 50),
    ("POSTER", 35),
    ("LAYOUT-LAYOUT", 35)]

    # A example font file.
    font_file = "./font_files/test_font.TTF"
    text_color = (255, 255 ,255)

    # A example background image.
    img_path = "./bk_image_folder/3AanCrYzXN0.png"

    # The iteration round of layout refine model.
    iteration_rounds = 30 

    ft_center = PutText2Image(font_file)
    bbox_size_array = np.zeros((len(sentences), 2))
    
    img = cv2.imread(img_path)
    width, height = img.shape[1], img.shape[0]
    img_size = (width, height)
    data_len = len(sentences)

    smooth_region_mask, regions, saliency_map = smooth_region_dectection(img)
    bbox_distrib_map = get_distrib_mask(smooth_region_mask)

    for i, text_info in enumerate(sentences):
        bbox_size_array[i] = ft_center.get_text_bbox_size(text=text_info[0], text_size=text_info[1])
    # initial layout, sampled by the maximum probability above the bbox_distrib_map.
    initial_bboxes = get_batch_text_region(bbox_distrib_map, bbox_size_array, img_size)
    print("initial_bboxes:", initial_bboxes.shape)
    initial_bbox_mask = get_bbox_mask(initial_bboxes, data_len)

    # The data used to refine the layout.
    initial_data = {"len_info": data_len, 
                 "shifted_mask": initial_bbox_mask.copy(),
                 "shifted_bbox": initial_bboxes.copy(),
                 "bbox_distrib_map": bbox_distrib_map.copy(),
                 "smooth_region_mask": smooth_region_mask.copy()}
    
    # The refined layout.
    refined_bboxes, refined_bbox_size, order = get_refine_bboxes(initial_data, iteration_rounds)
    print("refine_bboxes:", refined_bboxes.shape)
    refined_bbox_mask = get_bbox_mask(refined_bboxes[None, :], data_len)

    # scale the layout to the image size.
    refined_bboxes[:, (0, 2)] = refined_bboxes[:, (0, 2)] / STD_WIDTH * width
    refined_bboxes[:, (1, 3)] = refined_bboxes[:, (1, 3)] / STD_HEIGHT * height

    # show (salicy_map, predicted_layout_distribution, initial_layout) in a figure.
    add_sal_mask = np.zeros((height, width, 3))
    initial_bbox_mask = cv2.resize(initial_bbox_mask[0], (width, height))
    smooth_region_mask = cv2.resize(smooth_region_mask[0], (width, height))
    add_sal_mask[:, :, 2] = initial_bbox_mask / initial_bbox_mask.max()
    bbox_distrib_map = cv2.resize(bbox_distrib_map, (width, height))
    add_sal_mask[:, :, 1] = bbox_distrib_map / bbox_distrib_map.max()
    add_sal_mask[:, :, 0] = saliency_map / saliency_map.max() 
    add_sal_mask = cv2.resize(add_sal_mask, (width, height))
    cv2.imwrite("./display/initial_layout.jpg", add_sal_mask * 255)

    # show (salicy_map, predicted_layout_distribution, refined_layout) in a figure.
    add_sal_mask = np.zeros((height, width, 3))
    refined_bbox_mask = cv2.resize(refined_bbox_mask[0], (width, height))
    # smooth_region_mask = cv2.resize(smooth_region_mask[0], (width, height))
    add_sal_mask[:, :, 2] = refined_bbox_mask / refined_bbox_mask.max()
    bbox_distrib_map = cv2.resize(bbox_distrib_map, (width, height))
    add_sal_mask[:, :, 1] = bbox_distrib_map / bbox_distrib_map.max()
    add_sal_mask[:, :, 0] = saliency_map / saliency_map.max() 
    add_sal_mask = cv2.resize(add_sal_mask, (width, height))
    cv2.imwrite("./display/refined_layout.jpg", add_sal_mask * 255)

    # save the final poster picture.
    bk_img = img.copy()
    bk_img = bk_img.astype(np.uint8)
    for j in range(len(sentences)):
        text_position = (refined_bboxes[j][0], refined_bboxes[j][1])
        text, text_size = sentences[order[j]][0], sentences[order[j]][1]
        bk_img = ft_center.draw_text(bk_img, text_position, text, text_size, text_color)
    cv2.imwrite("./display/poster.jpg", bk_img)

