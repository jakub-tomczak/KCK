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
    # color between start=160 and stop=40
    #coloredMap = [[lerp(start = 160, stop = 40, value = i/rangeSpan) for i in row] for row in map]
    coloredMap = [[lerp(start = 120, stop = 0, value = (i-minH)/rangeSpan) for i in row] for row in map]
    return coloredMap


sunVector = [-200, -200, 500]
#col, row height
def calculateAngle(mainVector, bottomVector, rightVector):
    import math as m
    import numpy as np
    try:
        normalVector = np.cross(bottomVector - mainVector, rightVector - mainVector)
    except np.core._internal.AxisError:
        print("blad")
    angle = m.degrees(np.arccos(
        np.dot(normalVector, sunVector) / (np.linalg.norm(normalVector) * np.linalg.norm(sunVector))
    ))
    return angle
'''
    if (angle > 170):
        print("({x},{y})".format(x=row, y=col), "Min is ", minV, " Max is", maxV,
          "normal vector is {normal}".format(normal=normalVector), " angle is ", angle)

'''


def normalizeMap(map):
    import numpy as np


    normalizedMap = [[ calculateAngle(np.array([row, col, map[row][col]]) , np.array([row+1, col, map[row+1][col]]), np.array([row, col+1, map[row][col+1]]))
                      for col in range(0, len(map[row]) - 1)] for row in range(0, len(map) - 1)]
    return normalizedMap


def main():
    debug = False
    if(not debug):
        map, mapWidth, mapHeight, distance = importData("big.dem")
    else:
        map = [[1.0,1.0,1.0, 1.0], [1.0,2.0,1.5,1.0], [1.0,1.3,1.0, 1.0], [1.0, 1.2, 1.1, 1.0]]
        mapWidth = 4
        mapHeight = 4
        distance = 1
    coloredMap = colorMap(map)
    normalizedMap = normalizeMap(map)
    import matplotlib.pyplot as plt
    #represent array as list of RGB values
    img = [[hsv2rgb(coloredMap[row][col],
                    lerp(start=1,stop=.5, value=normalizedMap[row][col]/180),
                    lerp(start=1,stop=.5, value=normalizedMap[row][col]/180))
            for col in range(0, len(coloredMap[row]) - 1)] for row in range(0,len(coloredMap) - 1)]

    '''import numpy as np
    img = np.empty([mapHeight, mapWidth, 3])
    for row in range(0, mapHeight):
        for col in range(0, mapWidth):
            h = coloredMap[row][col]
            s = lerp(start=1,stop=.5, value=normalizedMap[row][col]/180)
            v = lerp(start=.9,stop=1, value=normalizedMap[row][col]/180)

            r, g, b = hsv2rgb(h,s,v)
            img[row][col][0] = r
            img[row][col][1] = g
            img[row][col][2] = b'''

    plt.imshow(img)
   # plt.show()
    if(debug):
        plt.savefig("map_d.pdf")
    else:
        plt.savefig("map.pdf")


if __name__ == "__main__":
    main()
    #normalizedMap()


