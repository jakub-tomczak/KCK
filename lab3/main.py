#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib.animation as animation
#matplotlib.use('Agg')                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from Xlib import display

from matplotlib import colors

def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    #rc('text', usetex=True)
    #rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400         # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)
    data = display.Display().screen().root.query_pointer()._data
    x, y = (data["root_x"], data["root_y"])
    w,h = (display.Display().screen()["width_in_pixels"], display.Display().screen()["height_in_pixels"])
    mean = (x / w + y / h) / 2

    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        if (gradient == gradient_hsv_custom):
            start = mean
            print("start is", start)
        else:
            start = 0

        for i, v in enumerate(np.linspace(start, 1, 1024)):
            if(start > 0):   #custom function
                img[:, i] = gradient_hsv_custom(start, v)
            else:
                img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)
    #fig.show()
    fig.savefig('gradients.pdf')

def hsv2rgb(h, s, v):
    import math
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    return r, g, b


def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    if v < 0.5:
        return lerpRGB((0, 1, 0), (0, 0, 1), 0, .5, v)
    else:
        return lerpRGB((0, 0, 1), (1, 0, 0), .5, 1, v)


def gradient_rgb_gbr_full(v):
    if(v > .75):
        return lerpRGB((1, 0, 1), (1, 0, 0), .75, 1, v)
    elif(v > .5):
        return lerpRGB((0, 0, 1), (1, 0, 1), .5, .75, v)
    elif(v > .25):
        return lerpRGB((0, 1, 1), (0, 0, 1), .25, .5, v)
    else:
        return lerpRGB((0, 1, 0), (0, 1, 1), 0, .25, v)


def gradient_rgb_wb_custom(v):
    if(v > .84):
        return lerpRGB((1, 0, 0), (0, 0, 0), .84, 1, v)
    elif(v > .70):
        return lerpRGB((0, 1, 0), (1, 0, 0), .70, .84, v)
    elif(v > .56):
        return lerpRGB((0, 1, 1), (0, 1, 0), .56, .70, v)
    elif(v > .42):
        return lerpRGB((0, 1, 1), (0, 1, 1), .42, 56, v)
    elif(v > .28):
        return lerpRGB((1, 0, 1), (0, 1, 1), .28, .42, v)
    elif(v > .14):
        return lerpRGB((1, 1, 0), (1, 0, 1), .14, .28, v)
    else:
        return lerpRGB((1, 1, 1), (1, 1, 0), 0, .14, v)


def gradient_hsv_bw(v):
    return hsv2rgb(0,0,v)


def gradient_hsv_gbr(v):
    return hsv2rgb(lerp(120, 360, v), 1, 1)

def gradient_hsv_unknown(v):
    return hsv2rgb(120-120*v,.5,1)

def gradient_hsv_custom(start, v):
    return hsv2rgb(360 * (start + (1 - v)), 1, 1)

def lerpRGB(color_in, color_out, start, end, step):
    r1, g1, b1 = color_in
    r2, g2, b2 = color_out
    percent = (step - start) / (end - start)
    return ( r1 + (r2-r1)*percent, g1 + (g2-g1)*percent, b1 + (b2-b1)*percent)

def lerp(start, stop, value):
    return start + (stop - start)*value

if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])
