#load file

from matplotlib.ticker import FuncFormatter

def getFiles(files):
    import csv
    for file in files:
        reader = (csv.reader(open(file, "r"), delimiter = ","))
        yield reader, file


def getData(content):
    firstLine = True
    xAxisData = []
    yAxisData = []
    lastColumn = []
    lastRow = []
    for row in content:
        if firstLine:
            firstLine = False
            continue
        lastRow = [float(i) for i in row]
        sum = 0
        xAxisData.append(int(row[1]))
        lastColumn.append(float(row[len(row) - 1]))
        for column in range(2, len(row)):
            sum += float(row[column])

        yAxisData.append(sum / (len(row) - 2))
    return xAxisData, yAxisData, lastColumn, lastRow[2:]


def prepareData(xAxisData):
    for i in range(len(xAxisData)):
        for j in range(len(xAxisData[i])):
            xAxisData[i][j] /= 1000

def thousands(x, pos):
    return '%1d' % (x*1e-3)

def percent(x, pos):
    return '%1d' % (x*1e+2)

def rowMean(row):
    return sum(row) / len(row)

def main():
    import glob
    import matplotlib.pyplot as graph
    files = glob.glob("*.csv")
    reader = getFiles(files)

    xAxisData = []
    yAxisData = []
    lastColumn = []
    lastRow = []
    for i in range(0, len(files)):
        content, fileName = next(reader)    #take content and fileName from generator

        print("File: {}".format(fileName))

        xAxisDataTemp, yAxisDataTemp, lastColumnTemp, lastRowTemp = getData(content)

        #add to global lists
        xAxisData.append(xAxisDataTemp)
        yAxisData.append(yAxisDataTemp)
        lastColumn.append(lastColumnTemp)
        lastRow.append(lastRowTemp)

    yLowerBound = round(min(yAxisData[0]),1)
    yUpperBound = 1.0
    plotsColors = ["g", "k", "m", "b", "r"]

    #first plot
    graph.figure(figsize=(5, 4))
    graph.ylabel("Odsetek wygranych gier")
    graph.xlabel("Rozegranych gier")
    graph.ylim(yLowerBound, yUpperBound)
    graph.xlim(0, 500000)
    graph.axes().tick_params(direction='in',top = True, right = True)
    for x, y, file, color in zip(xAxisData, yAxisData, files, plotsColors):
        graph.plot(x, y, color, label=file)
    graph.legend()

    graph.savefig("figures_0.pdf")



    #second plot
    graph.figure(figsize=(10, 7))
    secondPlot =  graph.subplot(121)
    #markers types
    markerType = ['v', 's', 'd', 'o', 'D' ]
    #plot data
    for x, y, file, marker, type in zip(xAxisData, yAxisData, files, plotsColors, markerType):
        secondPlot.plot(x, y, marker, label=file, linewidth = .9,  marker=type, markeredgecolor = 'k', markevery = 25, markersize = 6, ms = 4)
    secondPlot.set_ylabel("Odsetek wygranych gier [%]")
    secondPlot.set_xlabel("Rozegranych gier (x1000)")
    secondPlot.set_ylim(yLowerBound, yUpperBound)
    secondPlot.set_xlim(min(xAxisData[0]),max(xAxisData[0]))
    secondPlot.axes.tick_params(direction='in', top = True, right = True)
    secondPlot.grid(color = "grey", linestyle='dotted', linewidth=.5)


    #add top labels
    twiny = secondPlot.twiny()
    twiny.set_xlim(0,200)
    twiny.set_xticks([0,40,80,120,160,200])
    twiny.set_xlabel("Pokolenie")
    twiny.tick_params(direction="in")
    secondPlot.legend(shadow = True, loc=4, numpoints=2)

    #yAxis formatting
    formatter = FuncFormatter(percent)
    secondPlot.yaxis.set_major_formatter(formatter)
    #xAxis formatting
    formatter = FuncFormatter(thousands)
    secondPlot.xaxis.set_major_formatter(formatter)

    # boxplot
    boxplot = graph.subplot(122)
    data = []
    for x, file in zip(lastRow, files):
        data.append(x)
    box = boxplot.boxplot(data, 1, 'b+', vert=True, whis=1.5)
    #set box colors
    for boxBorder in box['boxes']:
        boxBorder.set_color('b')
    for means in box['means']:
        means.set_color('b')
    for whiskers in box['whiskers']:
        whiskers.set_color('b')
        whiskers.set_linestyle('dashed')

    boxplot.set_ylim(yLowerBound, yUpperBound)
    formatter = FuncFormatter(percent)
    boxplot.yaxis.set_major_formatter(formatter)
    boxplot.yaxis.tick_right()
    boxplot.yaxis.set_label_position("right")
    boxplot.xaxis.set_ticklabels([i for i in files], rotation =20)
    boxplot.grid(color = "grey", linestyle='dotted', linewidth=.5)
    boxplot.tick_params(direction="in")
    scattered = boxplot.scatter([i for i in range(1,6)], [rowMean(i) for i in lastRow])
    scattered.set_facecolors('b')
    graph.savefig("figures_1.pdf")
    graph.show()

if __name__ == "__main__":
    main()


