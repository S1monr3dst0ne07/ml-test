
import numpy as np
import math
from pyray import *
set_window_state(FLAG_WINDOW_RESIZABLE)
init_window(800, 450, "Hello")

import json
with open('comp.json', 'r') as f:
    net = json.load(f)

print(net)


dims = (len(net), max(len(l) for l in net))
print(dims)

max_weight = 0
for lay in net:
    for neu in lay:
        for wei in neu['weights']:
            max_weight = max(abs(wei), max_weight)

print(max_weight)


set_target_fps(1)

pos_cache = {}
frame_cache = {}

rad = 100

x_frame = np.ndarray((rad, rad))
y_frame = np.ndarray((rad, rad))
for y in range(rad):
    for x in range(rad):
        x_frame[y][x] = x / rad
        y_frame[y][x] = y / rad

frame_cache[(0, 0)] = x_frame
frame_cache[(0, 1)] = y_frame

def sigmoid(x):
    return np.where(
            x >= 0, # condition
            1 / (1 + np.exp(-x)), # For positive values
            np.exp(x) / (1 + np.exp(x)) # For negative values
    )
    
#render frames
for li_virt, lay in enumerate(net[1:]):
    li = li_virt + 1
    for ni, neu in enumerate(lay):
        frame = np.ndarray((rad, rad))
        
        for y in range(rad):
            for x in range(rad):
                sum = neu['bias']
                for i, weight in enumerate(neu['weights']): 
                    sum += weight * frame_cache[(li-1, i)][y][x]

                frame[y][x] = sigmoid(sum)

        frame_cache[(li, ni)] = frame



while not window_should_close():
    begin_drawing()
    clear_background(BLACK)
    space = rad * 2.4
    hspace = 140
    for li, layer in enumerate(net):
        for ni, neu in enumerate(layer):
            voffset = (dims[1] - len(layer)) / 2
            u = (li - (dims[0] // 2)) * (space + hspace)
            v = (ni - (dims[1] // 2) + voffset) * (space)
            x = int(u) + get_screen_width()  // 2
            y = int(v) + get_screen_height() // 2
            draw_circle_lines(x, y, rad, WHITE)
            pos_cache[(li, ni)] = (x, y)

            #preview
            frame = frame_cache[(li, ni)]
            for oy in range(rad):
                for ox in range(rad):
                    act = frame[oy][ox]
                    p = int(((act / 2) + 0.5) * 255)
                    color = (p, p, p, 255)
                    draw_pixel(
                        ox + x,
                        oy + y,
                        color
                    )


    for li in range(len(net)-1):
        cur = net[li]
        nxt = net[li+1]
        for src_i, src_neu in enumerate(cur):
            for dst_i, dst_neu in enumerate(nxt):
                w = dst_neu['weights'][src_i] / max_weight
                if abs(w) < 0.1: continue

                clamp = lambda x: x if x > 0 else 0
                c = Color(
                    int(clamp(-w) * 200) + 55,
                    0,
                    int(clamp(w) * 200) + 55,
                    int(abs(w) * 255)
                )
                start = pos_cache[(li, src_i)]
                end   = pos_cache[(li+1, dst_i)]
                draw_line_ex(
                    start,
                    end,
                    5.0,
                    c
                )


    end_drawing()

close_window()





