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

#get highest and lowest neighbour
#neigh array of size either 3x3 or 2x2
#middleElement is tuple (row,col)
def getMinMaxNeigh(neigh, middleElement):
    rows = len(neigh)
    cols = len(neigh[0])
    #min and max values are represented as tuples (value, row, col)
    minElement = (500, -1, -1)
    maxElement = (0, -1, -1)

    for row in range(0, rows):
        for col in range(0, cols):
            if(row == middleElement[0] and col == middleElement[1]):
                continue

            value = neigh[row][col]
            if(value < minElement[0]):
                minElement = (value, row, col)
            if(value > maxElement[0]):
                maxElement = (value, row, col)

    return (minElement, maxElement)
    #cases:
    #   minElement      maxElement
    #   higher          higher
    #   lower           lower
    #   lower           higher
    #   the same        the same

def normalizeMap(map):
    sunVector = [1,1,5]
    rows = len(map)
    cols = len(map[0])
    import numpy as np
    normalizedMap = np.empty([len(map), len(map[0])])
    maxAngle = 0
    for row in range(0,rows):
        for col in range(0, cols):
            neigh = []
            if(row == 0 or row == rows - 1):
                #top or bottom row
                neighRow = map[:2]      #we are in the top row
                middleRow = 0
                if(row == rows - 1):
                    neighRow = map[-2:]  #we are in the bottom row
                    middleRow = 1

                if(col == 0):           #top/bottom left
                    neigh = [ r[:2] for r in neighRow]
                    minV, maxV = getMinMaxNeigh(neigh, (middleRow, 0))

                    minV = (minV[0], middleRow*(row - ((minV[1] - 1) * (-1)) ) + (middleRow-1)*(-1)*minV[1], minV[2] )
                    maxV = (maxV[0], middleRow*(row - ((maxV[1] - 1) * (-1)) ) + (middleRow-1)*(-1)*maxV[1], maxV[2])

                elif(col == cols - 1):    #top/bottom right
                    neigh = [ r[-2:] for r in neighRow]
                    minV, maxV = getMinMaxNeigh(neigh, (middleRow, 1))
                    # row - middleRow*((minV[1]-1)*(-1)) for row == 0 returns row
                    # for last row decreases row by either 1 or 0
                    minV = (minV[0], middleRow*(row - ((minV[1] - 1) * (-1)) ) + (middleRow-1)*(-1)*minV[1], col + minV[2] - 1)
                    maxV = (maxV[0], middleRow*(row - ((maxV[1] - 1) * (-1)) ) + (middleRow-1)*(-1)*maxV[1], col + maxV[2] - 1)


                else:
                    neigh = [ r[col-1:col+2] for r in neighRow]
                    minV, maxV = getMinMaxNeigh(neigh, (middleRow, 1))
                    # row - middleRow*((minV[1]-1)*(-1)) for row == 0 returns row
                    # for last row decreases row by either 1 or 0
                    minV = (minV[0], middleRow*(row - ((minV[1] - 1) * (-1)) ) + (middleRow-1)*(-1)*minV[1],
                            col + minV[2] - 1)
                    maxV = (maxV[0], middleRow*(row - ((maxV[1] - 1) * (-1)) ) + (middleRow-1)*(-1)*maxV[1],
                            col + maxV[2] - 1)


            elif(col == 0 ):
                #left column
                neighRow = map[row-1:row+2]  # we are in the top row
                neigh = [r[:2] for r in neighRow]
                minV, maxV = getMinMaxNeigh(neigh, (1, 0))
                # row - middleRow*((minV[1]-1)*(-1)) for row == 0 returns row
                # for last row decreases row by either 1 or 0
                minV = (minV[0], row + minV[1] - 1, col + minV[2])
                maxV = (maxV[0], row + maxV[1] - 1, col + maxV[2])



            elif(col == cols - 1):
                #right column
                neighRow = map[row-1:row+2]  # we are in the top row
                neigh = [r[-2:] for r in neighRow]
                minV, maxV = getMinMaxNeigh(neigh, (1, 1))
                # col - (maxV[0]-1)*(-1) -> col - 1 if maxV[0] == 0 or col - 0 if maxV[0] == 1
                minV = (minV[0], row + minV[1] - 1, col - (minV[2]-1)*(-1))
                maxV = (maxV[0], row + maxV[1] - 1, col - (maxV[2]-1)*(-1))

            else:
                neighRow = map[row - 1: row + 2]            #get 3 rows
                neigh = [ r[col-1:col+2] for r in neighRow] #get 3 columns
                minV, maxV = getMinMaxNeigh(neigh, (1,1))
                minV = (minV[0], row + minV[1] - 1, col + minV[2] - 1)
                maxV = (maxV[0], row + maxV[1] - 1, col + maxV[2] - 1)

            if(minV[0] < map[row][col] and maxV[0] < map[row][col] or minV[0] < map[row][col] and maxV[0] < map[row][col]
               or maxV[0]-minV[0] < 5):
                angle = 0
            else:
                mainVector = np.array([col, row, map[row][col]])
                firstVector = np.array([minV[2], minV[1], minV[0]])
                secondVector = np.array([maxV[2], maxV[1], maxV[0]])

                import math as m
                normalVector = np.cross(firstVector - mainVector, secondVector - mainVector)
                angle = m.degrees(np.arccos(
                    np.dot(normalVector, sunVector) / (np.linalg.norm(normalVector) * np.linalg.norm(sunVector))
                ))

            if(angle > maxAngle):
                maxAngle = angle
            angle = angle / 2
            if(angle>170):
                print("({x},{y})".format(x = row, y = col),"Min is ", minV, " Max is", maxV, "normal vector is {normal}".format(normal = normalVector), " angle is ", angle)

            normalizedMap[row][col] = angle
            #print(row, col, minV, maxV)
    print(maxAngle)


    return normalizedMap


def main():
    debug = not True
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
    #img = [[hsv2rgb(coloredMap[row][col], lerp(start=1,stop=.2, value=normalizedMap[row][col]/180), lerp(start=1,stop=.5,
    # value=normalizedMap[row][col]/180)) for col in range(0, len(coloredMap[row] ))] for row in range(0,len(coloredMap))]
    import numpy as np
    img = np.empty([mapHeight, mapWidth, 3])
    for row in range(0, mapHeight):
        for col in range(0, mapWidth):
            h = coloredMap[row][col]
            s = lerp(start=1,stop=.7, value=normalizedMap[row][col]/180)
            v = lerp(start=.9,stop=1, value=normalizedMap[row][col]/180)

            r, g, b = hsv2rgb(h,s,v)
            img[row][col][0] = r
            img[row][col][1] = g
            img[row][col][2] = b

    plt.imshow(img)
    plt.show()
    if(debug):
        plt.savefig("map_d.pdf")
    else:
        plt.savefig("map.pdf")


if __name__ == "__main__":
    main()
    #normalizedMap()


