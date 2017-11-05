from __future__ import division
from pylab import *
from skimage import io, filters
import skimage.morphology as mp
import skimage.measure as measure
from skimage.color import rgb2gray

import numpy as np
from ipykernel.pylab.backend_inline import flush_figures

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
    return [r*255, g*255, b*255]
def lerp(start, stop, value):
    return start + (stop - start)*value

def contourDetector(image, threshold = .15):
    gray = rgb2gray(image)
    sobelImage = filters.sobel(gray)
    thresold = mp.dilation(colorThreshold(sobelImage, getMax(sobelImage)*threshold))
    #out = measure.label(thresold, background=0, neighbors=8, return_num=True)
    return thresold

def processAll(threshold = .15):
    lastFileIndex = 20
    images = [readImage("samolot%02d" % i) for i in range(0, lastFileIndex+ 1)]
    contours = [contourDetector(image) for image in images]
    displaySaveImage(contours, "planes_t{}.png".format(threshold))


def processOne(number):
    filename = "samolot%02d" % number
    image = readImage(filename)
    contour = contourDetector(image)
    labeled, num = measure.label(contour, background=0, neighbors=8, return_num=True)
    colored = np.zeros([len(labeled), len(labeled[0]), 3])
    imSize = len(labeled) * len(labeled[0])

    unique, counts = np.unique(labeled, return_counts=True)
    values = dict(zip(unique, counts))
    #print(values)
    #print(len(labeled), len(labeled[0]))
    labeled = [[ (values[value] > imSize*0.0005)*value for value in row] for row in labeled]

    #unique, counts = np.unique(labeled, return_counts=True)
    #values = dict(zip(unique, counts))
    #print(values)


    #return 1
    for row in range(0, len(labeled)):
        for i in range(0, len(labeled[row])):
            if labeled[row][i] > 0:
                image[row][i] = hsv2rgb(lerp(0, 360, labeled[row][i]/num), 1, 1)

    #imColor = [[rank[labeled[row][value]] for value in range(0, len(row))] for row in range(0, len(labeled))]
    #print("Number of labels {}".format(num))

    displaySaveImage([image], "one/"+filename, resolution=100)

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
    #plt.show()
def main():
    #processAll()
    processOne(23)
    #[processOne(num) for num in range(0,21)]
if __name__ == "__main__":
    main()
