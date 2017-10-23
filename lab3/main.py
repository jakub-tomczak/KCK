#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib
matplotlib.use('Agg')                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

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


    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')


def hsv2rgb(h, s, v):
    C = v * s
    X = C * (1 - abs((h/60)%2 - 1))
    m = v - C
    r_0 = 0
    g_0 = 0
    b_0 = 0
    if h >= 300:
        r_0 = C
        g_0 = 0
        b_0 = X
    elif h >=240:
        r_0 = X
        g_0 = 0
        b_0 = C
    elif h >=180:
        r_0 = 0
        g_0 = X
        b_0 = C
    elif h >=120:
        r_0 = 0
        g_0 = C
        b_0 = X
    elif h >=60:
        r_0 = X
        g_0 = 0
        b_0 = C
    else:
        r_0 = C
        g_0 = X
        b_0 = 0
    return ((r_0+m)*255, (g_0+m)*255, (b_0+m)*255)

def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    if v < 0.5:
        return (0, 2*(v*(-1) + 1), 2*v)
    else:
        return ((v-.5)*2, 0, 2 + (v*-2))


def gradient_rgb_gbr_full(v):
    if v < .5:
        return (0,  min(2*(v*(-1) + 1)*2, 1-4*(v*(-1) + 1)), 2*v)
    else:
        return (0,0,0)


def gradient_rgb_wb_custom(v):
    #TODO
    return (0, 0, 0)


def gradient_hsv_bw(v):
    #TODO
    return hsv2rgb(v, v, v)


def gradient_hsv_gbr(v):
    #TODO
    return hsv2rgb(0, 0, 0)

def gradient_hsv_unknown(v):
    #TODO
    return hsv2rgb(0, 0, 0)


def gradient_hsv_custom(v):
    #TODO
    return hsv2rgb(0, 0, 0)


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])