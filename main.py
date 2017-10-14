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

    #first plot
    graph.figure(figsize=(5, 4))

    graph.ylabel("Odsetek wygranych gier")
    graph.xlabel("Rozegranych gier")
    graph.ylim(yLowerBound, yUpperBound)
    graph.xlim(0, 500000)
    graph.legend(shadow = True)

    for x, y, file in zip(xAxisData, yAxisData, files):
        graph.plot(x, y, label=file)



    #second plot
    graph.figure(figsize=(10, 7))
    secondPlot =  graph.subplot(121)
    #markers styles and types
    markerStyles = ["g", "k", "m", "b", "r"]
    markerType = ['v', 's', 'd', 'o', 'D' ]

    #plot data
    for x, y, file, marker, type in zip(xAxisData, yAxisData, files, markerStyles, markerType):
        secondPlot.plot(x, y, marker, label=file, linewidth = .9,  marker=type, markevery = 25, ms = 4)
    secondPlot.set_ylabel("Odsetek wygranych gier [%]")
    secondPlot.set_xlabel("Rozegranych gier (x1000)")
    secondPlot.set_ylim(yLowerBound, yUpperBound)
    secondPlot.set_xlim(min(xAxisData[0]),max(xAxisData[0]))

    #add top labels
    twiny = secondPlot.twiny()
    twiny.set_xlim(0,200)
    twiny.set_xticks([0,40,80,120,160,200])
    twiny.set_xlabel("Pokolenie")
    secondPlot.legend(shadow = True, loc=4)

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
    boxplot.boxplot(data, 1, 'b+', vert=True,
                    showmeans=True, whis=1.5)
    boxplot.set_ylim(yLowerBound, yUpperBound)
    formatter = FuncFormatter(percent)
    boxplot.yaxis.set_major_formatter(formatter)
    boxplot.yaxis.tick_right()
    boxplot.yaxis.set_label_position("right")
    boxplot.grid(color = "tab:grey", linestyle='dashed', linewidth=1)
    boxplot.scatter([i for i in range(1,6)], [rowMean(i) for i in lastRow])

    graph.show()

if __name__ == "__main__":
    main()


