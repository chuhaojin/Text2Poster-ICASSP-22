import freetype
import copy
import os
import requests
import jsonlines
import numpy as np

# 改为居中放置字体

class PutText2Image(object):
    def __init__(self, ttf):
        self.ttf = ttf
        self._face = freetype.Face(self.ttf)

    def get_text_bbox_size(self, text, text_size):
        x_pos = 0
        y_pos = 0
        y_pos = text_size // 2
        high_y = 0
        low_y = 1000
        self._face.set_char_size(text_size * 64)
        for cur_char in text:
            self._face.load_char(cur_char)
            slot = self._face.glyph
            rows, width = slot.bitmap.rows, slot.bitmap.width
            t_width = slot.bitmap.width
            if ord(cur_char) == ord(" "):
                t_width = slot.advance.x >> 6
#                 t_width = slot.bitmap.width + text_size//6
            else:
                t_width = min(slot.bitmap.width + text_size//6, slot.advance.x >> 6)
            x_pos += t_width
            if slot.bitmap_top > high_y:
                high_y = slot.bitmap_top
            if (slot.bitmap_top - slot.bitmap.rows) < low_y:
                low_y = slot.bitmap_top - slot.bitmap.rows
#                 print("top:", slot.bitmap_top, "height:", slot.bitmap.rows)
#             print(high_y, low_y)
        y_pos = high_y - low_y
        bbox_width, bbox_height = x_pos, y_pos
        bbox_size = (bbox_width, text_size)
#         print("bbox_size:", bbox_size)
        return bbox_size
    
    def draw_text_center(self, image, pos, text, text_size, text_color):
        bbox_size = self.get_text_bbox_size(text, text_size)
        left = pos[0] - bbox_size[0] // 2
        up = pos[1] - bbox_size[1] // 2
        return self.draw_text(image, (left, up), text, text_size, text_color)
    
    def draw_text(self, image, pos, text, text_size, text_color):
        '''
          draw chinese(or not) text with ttf
          :param image:  image(numpy.ndarray) to draw text
          :param pos:  where to draw text, (the text's center)
          :param text:  the context, for chinese should be unicode type
          :param text_size: text size
          :param text_color:text color
          :return:   image
          '''
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0
        ypos = int(ascender)
        text = text
        img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color, text_size)
        
        return img

    def draw_string(self, img, x_pos, y_pos, text, color, text_size):
        '''
        draw string
          :param x_pos: text x-postion on img
          :param y_pos: text y-postion on img
          :param text: text (unicode)
          :param color: text color
          :return:  image
          '''
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6 # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.0 * 0x10000), 
                                 int(0.0 * 0x10000), int(1.0 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)
            if ord(cur_char) == ord(" "):
                t_width = slot.advance.x
#                 t_width = (slot.bitmap.width + text_size//6) << 6
            else:
                t_width = min((slot.bitmap.width + text_size//6) << 6, slot.advance.x)
            pen.x += (t_width)
            prev_char = cur_char
#             print(slot.advance.x, bitmap.rows, bitmap.width * 64)
        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        '''
          draw each char
          :param bitmap: bitmap
          :param pen: pen
          :param color: pen color e.g.(0,0,255) - red
          :return:  image
        '''
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        color = np.array(color).astype(np.uint8)
        cols = bitmap.width
        rows = bitmap.rows
        glyph_pixels = bitmap.buffer
        glyph_pixels = np.array(glyph_pixels, dtype = np.uint8)
        try:
            glyph_pixels = glyph_pixels.reshape([rows, cols, 1])
        except:
            pass
        if img.shape[2] > 1:
            text_mask = np.concatenate([glyph_pixels for _ in range(img.shape[2])],axis = 2)
#         glyph_pixels = glyph_pixels[:, :, np.newaxis,]
        bk_region = img[y_pos: y_pos + rows, x_pos: x_pos + cols].copy()
        h, w = bk_region.shape[0], bk_region.shape[1]
        text_mask = text_mask[:h, :w]
        fusion_region = np.where(text_mask !=0, color, bk_region)
#         fusion_region = bk_region * (~text_mask) + text_mask * color
        img[y_pos: y_pos + h, x_pos: x_pos + w] = fusion_region

        return

def load_jsonl(path):
    data = []
    with jsonlines.open(path, mode = "r") as reader:
        for obj in reader:
            data.append(obj)
    return data

def get_text_emb(text):
    url = "http://buling.wudaoai.cn/text_query"
    params = {"text":text}
    r = requests.get(url, params=params)
    emb_data = r.json()
    emb = np.array(emb_data["embedding"])
    return emb

def get_img_emb(img_path):
    url = "http://buling.wudaoai.cn/image_query"
    files = {"image": open(img_path, "rb")}
    r = requests.post(url, files=files)
    emb_data = r.json()
    emb = np.array(emb_data["embedding"])
    return emb
