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

def plot_hist(img):
    img = img_as_ubyte(img)
    histo, x = np.histogram(img, range(0, 10), density=True)
    plot(histo)
    xlim(0, 10)

def showHist(image, filename=""):
    fig = figure(figsize=(30, 30))
    subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    subplot(1, 2, 2)
    plot_hist(image)

    if len(filename) > 0:
        fig.savefig("out/hist_{}.png".format(filename))
    else:
        plt.show()

def getMean(desc, image):
    mean = np.mean(image)
    print("{} image mean = {}".format(desc,mean))
    return mean

def getMax(desc, image):
    maxV = max(np.max(image, axis=1))
    print("{} image max = {}".format(desc, maxV))
    return maxV

def sobel(img, gray=True):
    if gray:
        return filters.sobel(rgb2gray(img))
    else:
        return filters.sobel(img)

def colorThreshold(image, t):
    processed = (image > t) * 1
    flush_figures()
    return processed

def readImage(name):
    return  io.imread("data/{img}.jpg".format(img = name))

def displaySaveImage(imgs, filename = "planes_bin.png"):
    fig = figure(figsize=(20,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 +1)
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig("out/"+filename, dpi=500)

def imageProcessor(img, filename="planes_bin.png"):
    img = rgb2gray(img)**.4

    sobelImage = sobel(img, False)
    #showHist(sobelImage, "sobel_{}".format(number))
    showHist(sobelImage)
    mean = getMean("sobel_mean_", sobelImage)

    maxV = getMax("sobel_max_", sobelImage)

    print("maxV", maxV, " mean ", mean)
    binary = colorThreshold(sobelImage, maxV*.2)

    sobo2 = sobel(binary)
    showHist(binary)
    getMean("binary", binary)
    #showHist(binary,"binary_{}".format(number))

    showHist(sobo2)

    displaySaveImage([sobo2], filename)


def processAll():
    planesCount = 20
    images = [readImage("samolot%02d" % i) for i in range(0, planesCount+ 1)]
    greyImgs = [rgb2gray(img)**.8 for img in images]
    '''cannys = [ski.feature.canny(rgb2gray(image)) for image in images]
displaySaveImage(cannys)
return 1'''
    sobelImages = [sobel(image, False) for image in greyImgs]
    binary = [colorThreshold(image, getMax("zdjecie", image)*.2) for image in sobelImages]

  #  processStep2 = [ski.feature.canny(image, sigma=1) for image in binary]
    displaySaveImage(binary)


def processOne(number):
    filename = "samolot%02d" % number
    print("processing " + filename)
    image = [readImage(filename)]
    showHist(rgb2gray(image[0])**.2)
    canny = ski.feature.canny(rgb2gray(image[0]))
    showHist(canny)
    return 1
    img = rgb2gray(image[0])**2

    sobelImage = sobel(img, False)
    #showHist(sobelImage, "sobel_{}".format(number))
    #showHist(sobelImage)
    mean = getMean("sobel_mean_", sobelImage)

    maxV = getMax("sobel_max_", sobelImage)

    print("maxV", maxV, " mean ", mean)
    binary = colorThreshold(sobelImage, maxV*.1)

    sobo2 = sobel(binary)
    showHist(binary)
    getMean("binary", binary)
    showHist(binary,"binary_{}".format(number))

    showHist(sobo2)

    canny = ski.feature.canny(binary, sigma=1)
    showHist(canny)
    #displaySaveImage([binary], filename)


def main():
    debug = False

    if debug:
        try:
            processOne(19)
            #[processOne(i) for i in [2,6,13,14,15,20]]
        except FileNotFoundError:
            print("Podany plik nie istnieje")

    else:
        processAll()
    print("done")

def contour():
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage import measure
    img = readImage("samolot20")
    img = rgb2gray(img)
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(img, 0.7)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
if __name__ == "__main__":
    #contour()
    main()