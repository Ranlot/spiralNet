import matplotlib.pyplot as plt
from .generalUtils import *
import numpy as np

markerColor_dict = {0 : 'gold', 1 : 'blue', 2 : 'darkred', 3 : 'green', 4 : 'purple'}
bgColor_dict = {0 : 'yellow', 1 : 'cyan', 2 : 'tomato', 3 : 'lime', 4 : 'fuchsia'}

def plotSpiral(spiralData):
    X = extractElem(0, spiralData)
    y = extractElem(1, spiralData)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.axis('equal')
    plt.savefig

def colorGetter(colorDict):
    return lambda colorKeys: [colorDict.get(colorKey, 'black') for colorKey in colorKeys]

markerColor, bgColor = [colorGetter(colorDict) for colorDict in [markerColor_dict, bgColor_dict]]

def plotData(testingGrid, gridResults, positionData, labelData, plotDir):
    f, ax = plt.subplots()
    ax.scatter(testingGrid[:, 0], testingGrid[:, 1], c=bgColor(gridResults), s=10, cmap=plt.cm.Spectral, alpha=0.3)
    ax.scatter(positionData[:, 0], positionData[:, 1], c=markerColor(np.argmax(labelData, 1)), s=40, cmap=plt.cm.Spectral, edgecolor='black', marker='s', linewidth=1)
    ax.set_title('Decision boundaries @ input space')
    plt.savefig('%s/inputData.png' % plotDir)

def lossPlotter(numEpochs, lossValues, plotDir):
    f, ax = plt.subplots()
    ax.plot(range(1, 1+numEpochs), lossValues, '-', lw=3, c='k')
    ax.set_title(r'Cross entropy loss vs. epochs (log 5 $\approx$ %.2f)' % (np.log(5)))
    ax.grid()
    plt.savefig('%s/loss.png' % plotDir)
 
