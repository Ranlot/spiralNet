from operator import itemgetter
import numpy as np

def extractElem(elemToExtract, data):
    return np.concatenate(list(map(itemgetter(elemToExtract), data)))

def makeGrid(xmin, xmax, ymin, ymax):
    gridData = np.linspace(xmin, xmax, 500)
    gridX, gridY = np.meshgrid(gridData, gridData)
    return np.dstack((gridX.flatten(), gridY.flatten()))[0]

