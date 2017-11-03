#%matplotlib inline
from __future__ import division
from pylab import *
import skimage as ski
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

def plot_hist(img):
    img = img_as_ubyte(img)
    histo, x = np.histogram(img, range(0, 10), density=True)
    plot(histo)
    xlim(0, 10)

def showHist(image, filename=""):
    fig = figure(figsize=(15, 5))
    subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    subplot(1, 2, 2)
    plot_hist(image)

    if len(filename) > 0:
        fig.savefig("out/hist_{}.png".format(filename))
    else:
        fig.show()

def getMean(desc, image):
    mean = np.mean(image)
    print("{} image mean = {}".format(desc,mean))
    return mean

def getMax(desc, image):
    maxV = max(np.max(image, axis=1))
    print("{} image max = {}".format(desc, maxV))
    return maxV

def sobel(img):
    return filters.sobel(rgb2gray(img))

def colorThreshold(image, t):
    processed = (image > t) * 1
    flush_figures()
    return processed

def readImage(name):
    return  io.imread("data/{img}.jpg".format(img = name))

def displaySaveImage(imgs, filename = "planes_bin.png"):
    fig = figure(figsize=(10,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 )
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig("out/"+filename, dpi=300)

def processAll():
    planesCount = 19
    images = [readImage("samolot%02d" % i) for i in range(0, planesCount+ 1)]
    sobelImages = [sobel(image) for image in images]
    binary = [colorThreshold(image, getMax("zdjecie", image)*.3) for image in sobelImages]
    displaySaveImage(images + binary)

def processOne(number):
    filename = "samolot%02d" % number
    print("processing " + filename)
    image = [readImage(filename)]


    sobelImage = sobel(image[0])
    showHist(sobelImage, "sobel_{}".format(number))
    maxV = getMax("sobel", sobelImage)


    binary = colorThreshold(sobelImage, maxV*.25)
    getMean("binary", binary)
    showHist(binary,"binary_{}".format(number))


    displaySaveImage([binary], filename)


def main():
    debug = False

    if debug:
        try:
            processOne(15)
        except FileNotFoundError:
            print("Podany plik nie istnieje")

    else:
        processAll()



if __name__ == "__main__":
    main()