from __future__ import division
from pylab import *
from skimage import io, filters
import skimage.morphology as mp
from skimage.color import rgb2gray

import numpy as np
from ipykernel.pylab.backend_inline import flush_figures

def contourDetector(image, threshold = .15):
    gray = rgb2gray(image)
    sobelImage = filters.sobel(gray)
    thresold = mp.dilation(colorThreshold(sobelImage, getMax(sobelImage)*threshold))
    return thresold

def processAll(threshold = .15):
    lastFileIndex = 20
    images = [readImage("samolot%02d" % i) for i in range(0, lastFileIndex+ 1)]
    contours = [contourDetector(image) for image in images]
    displaySaveImage(contours, "planes_t{}.png".format(threshold))

def processOne(number):
    filename = "samolot%02d" % number
    image = readImage(filename)
    displaySaveImage([contourDetector(image)], filename)

def colorThreshold(image, t):
    processed = (image > t) * 1
    flush_figures()
    return processed

def getMax(image):
    maxV = max(np.max(image, axis=1))
    return maxV

def readImage(name):
    return  io.imread("data/{img}.jpg".format(img = name))

def displaySaveImage(imgs, filename = "planes.png", resolution = 500):
    fig = figure(figsize=(20,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 +1)
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig("out/"+filename, dpi=resolution)

def main():
    processAll()

if __name__ == "__main__":
    main()
