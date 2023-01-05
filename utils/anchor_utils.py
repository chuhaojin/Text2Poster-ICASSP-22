from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.boxes import nms
import numpy as np
import torch


sizes = (32, 64, 128)
aspect_ratios = (0.2, 0.1, 0.333, 0.5, 1.0, 2.0)

top_n = 30

iou_threshold = 0.00

test_id = 2045

grid_sizes = [(20, 20)]
strides = [(20, 15)]

anchor_generator = AnchorGenerator(sizes=(sizes, ),
                                   aspect_ratios=(aspect_ratios, ))

anchor_generator.set_cell_anchors(dtype=torch.float32, device=torch.device("cpu"))
test_anchor = anchor_generator.grid_anchors(grid_sizes, strides)

inbound_bbox_list = []
np_test_anchor = test_anchor[0].numpy().astype(np.int32)

for bbox in np_test_anchor:
    if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] < 300 and bbox[3] < 400:
        inbound_bbox_list.append(bbox)

inbound_bbox_list = np.array(inbound_bbox_list)
inbound_bbox = torch.tensor(inbound_bbox_list)
# print("inbound_bbox:", inbound_bbox.shape)


def get_candidates_region(saliency_map, inbound_bbox = inbound_bbox, threshold = 0.7):
    '''
    saliency_map: (height, width)
    '''
    mean_sals = np.zeros(shape = (len(inbound_bbox), ))
    mean = saliency_map.mean()

    for i, bbox in enumerate(inbound_bbox):
        # 不仅要saliency均值最小，同时希望改map尽量大一些（trade-off）
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        s = h * w
        alpha = 1 / (s * mean + 0.0000001)
        mean_sals[i] = saliency_map[bbox[1]:bbox[3], bbox[0]: bbox[2]].mean() + alpha

    bboxes_ids = nms(boxes = inbound_bbox.clone().detach().to(torch.float64), 
                     scores = torch.tensor(-mean_sals).clone().detach(), iou_threshold = iou_threshold)

    filtered_bbox = inbound_bbox[bboxes_ids]
    filtered_scores = mean_sals[bboxes_ids]

    filtered_bbox = filtered_bbox[filtered_scores <= (mean * 2 * threshold)]
    filtered_scores = filtered_scores[filtered_scores <= (mean * 2 * threshold)]

    min_bboxs_n = filtered_bbox[:top_n]
    filtered_scores_n = filtered_scores[:top_n]
    return min_bboxs_n, filtered_scores_n


def get_text_region(layouts_map, bbox_size, top_n = 1):
    '''
    layouts_map: (height, width)
    bbox_size: the text bbox's size, (text_width, text_height)
    '''
    
    text_width, text_height = bbox_size[0], bbox_size[1]
    height, width = layouts_map.shape[0], layouts_map.shape[1]
    cand_text_bbox = np.array([[i, j, i + text_width, j + text_height] 
                               for i in range(width - text_width)
                               for j in range(height - text_height)])
    mean_sals = np.zeros(shape = (len(cand_text_bbox), ))
    mean = layouts_map.mean()

    for i, bbox in enumerate(cand_text_bbox):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        s = h * w
#         alpha = 1 / (s * mean + 0.0000001)
        mean_sals[i] = layouts_map[bbox[1]:bbox[3], bbox[0]: bbox[2]].mean()

    bboxes_ids = nms(boxes = torch.tensor(cand_text_bbox, dtype = torch.float64).clone().detach(), 
                     scores = torch.tensor(mean_sals).clone().detach(), iou_threshold = iou_threshold)

    filtered_bbox = cand_text_bbox[bboxes_ids]
    filtered_scores = mean_sals[bboxes_ids]

#     filtered_bbox = filtered_bbox[filtered_scores >= (mean / 1.4)]
#     filtered_scores = filtered_scores[filtered_scores >= (mean / 1.4)]

    min_bboxs_n = filtered_bbox[:top_n]
    filtered_scores_n = filtered_scores[:top_n]
    return min_bboxs_n, filtered_scores_n


if __name__ == "__main__":
    import cv2
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    rand_img = np.random.random((400, 300, 3)) * 255
    rand_img = rand_img.astype(np.uint8)
    success_info, sal_img = saliency.computeSaliency(rand_img)
    cand_bboxes, cand_scores = get_candidates_region(sal_img)
    print(cand_bboxes)
    print(cand_scores)
