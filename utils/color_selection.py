# 颜色部分

import numpy as np
import random

def color_rotation_random(rgb_color, rotation_degree = 90):
    if rotation_degree > 180 or rotation_degree < 0:
        raise ValueError("rotation_degree should <= 180 and >= 0.")
    r, g, b = rgb_color[0], rgb_color[1], rgb_color[2]
    sum_c_abs = int(rotation_degree * 768 // 180)
    max_c_abs = max(255 - r, r) + max(255 - g, g) + max(255 - b, b) - 3
    sum_c_abs = min(sum_c_abs, max_c_abs)
    orgin_r, orgin_g, orgin_b = rgb_color[0], rgb_color[1], rgb_color[2]
    next_c_abs = sum_c_abs
    while sum_c_abs != 0:
        r_range = np.array((max(0, r - next_c_abs), min(255, r + next_c_abs))) - r
        if random.random() > 0.1:
            r_shift = np.random.randint(low = r_range[0], high = r_range[1] + 1)
        else:
            r_shift = np.random.choice(r_range)
        r = r + r_shift
        next_c_abs = sum_c_abs - np.abs(orgin_r - r) - np.abs(orgin_g - g) - np.abs(orgin_b - b)
        if next_c_abs == 0:
            break
        g_range = np.array((max(0, g - next_c_abs), min(255, g + next_c_abs))) - g
        if random.random() > 0.1:
            g_shift = np.random.randint(low = g_range[0], high = g_range[1] + 1)
        else:
            g_shift = np.random.choice(g_range)
        g = g + g_shift
        next_c_abs = sum_c_abs - np.abs(orgin_r - r) - np.abs(orgin_g - g) - np.abs(orgin_b - b)
        if next_c_abs == 0:
            break
        b_range = np.array((max(0, b - next_c_abs), min(255, b + next_c_abs))) - b
        if random.random() > 0.1:
            b_shift = np.random.randint(low = b_range[0], high = b_range[1] + 1)
        else:
            b_shift = np.random.choice(b_range)
        b = b + b_shift
        next_c_abs = sum_c_abs - np.abs(orgin_r - r) - np.abs(orgin_g - g) - np.abs(orgin_b - b)
    return np.array((r, g, b))


if __name__ == "__main__":
    import time
    r, g, b = 90, 80, 0
    back_color = np.array((r, g, b))
    color_map = np.zeros(shape = (1000, 1200, 3), dtype = np.uint8)
    color_map = color_map + back_color
    tme_a = time.time()
    for i in range(5):
        new_color = color_rotation_random(back_color, 30)
        print(new_color)
        color_map[:, (i * 2) * 100: (i * 2 + 1) * 100] = new_color
    tme_b = time.time()
    print(tme_b - tme_a)
