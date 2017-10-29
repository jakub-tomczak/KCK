def importData(file):
    with open(file) as file:
        mapData = file.read().splitlines()
    mapData = [line.split(' ') for line in mapData]
    mapHeight = int(mapData[0][0])
    mapWidth = int(mapData[0][1])
    distance = int(mapData[0][2])
    del mapData[0] #remove first row
    for row in range(len(mapData)):
        del mapData[row][-1]    #removes empty character
        mapData[row] = [float(i) for i in mapData[row]]

    return (mapData, mapHeight, mapWidth, distance)


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
    return [r, g, b]
def lerp(start, stop, value):
    return start + (stop - start)*value

def colorMap(map):
    import numpy as np
    maxH = max(np.max(map, axis=1))
    minH = min(np.min(map, axis=1))
    rangeSpan = maxH - minH
    #represents color in HSV
    # color between 0 - red and 120 - green
    coloredMap = [[lerp(160,40, i/rangeSpan) for i in row] for row in map]
    return coloredMap

if __name__ == "__main__":
    map, mapWidth, mapHeight, distance = importData("big.dem")
    coloredMap = colorMap(map)

    import matplotlib.pyplot as plt
    #represent array as list of RGB values
    img = [[hsv2rgb(coloredMap[row][col], 1, 1) for col in range(0, len(coloredMap[row] ))] for row in range(0,len(coloredMap))]

    plt.imshow(img)
    plt.show()

