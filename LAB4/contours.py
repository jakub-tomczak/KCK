#%matplotlib inline
from __future__ import division
from pylab import *
import skimage as ski
from skimage import feature
from skimage import data, io, filters, exposure
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array
from IPython.display import display
from ipywidgets import interact, interactive, fixed
from ipywidgets import *
from ipykernel.pylab.backend_inline import flush_figures



def processAll(gamma = 1, threshold = .15):
    planesCount = 20
    images = [rgb2gray(readImage("samolot%02d" % i)) for i in range(0, planesCount+ 1)]
    sobelImages = [filters.sobel(image) for image in images]
    thresold = [mp.erosion( mp.dilation(colorThreshold(image, getMax(image)*threshold))) for image in sobelImages]
    displaySaveImage(thresold, "planes_g{}_t{}_dilation_er_debug.png".format(gamma, threshold))


def colorThreshold(image, t):
    processed = (image > t) * 1
    flush_figures()
    return processed

def getMax(image):
    maxV = max(np.max(image, axis=1))
    return maxV

def readImage(name):
    return  io.imread("data/{img}.jpg".format(img = name))

def displaySaveImage(imgs, filename = "planes.png"):
    fig = figure(figsize=(20,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 +1)
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig("out/"+filename, dpi=500)

def main():
    processAll()

if __name__ == "__main__":
    main()
