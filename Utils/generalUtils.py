from operator import itemgetter
import numpy as np

def extractElem(elemToExtract, data):
    return np.concatenate(list(map(itemgetter(elemToExtract), data)))

def makeGrid(dataSet, num=500):
    def extremePadder(_):
        padRatio = 0.06
        minVal, maxVal = _.min(), _.max()
        return minVal - padRatio * np.abs(minVal), maxVal + padRatio * np.abs(maxVal)
    gridMaker = lambda axis: np.linspace(num=num, *extremePadder(dataSet[:, axis]))
    return np.dstack(map(lambda _: _.flatten(), np.meshgrid(*map(gridMaker, [0, 1]))))[0]
