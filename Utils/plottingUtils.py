import tensorflow as tf
import scipy.stats as st
import matplotlib.pyplot as plt
from .generalUtils import *
import numpy as np
import os
import pandas as pd

markerColor_dict = {0 : 'gold', 1 : 'blue', 2 : 'darkred', 3 : 'green', 4 : 'purple', 5: 'darkorange', 6: 'gray'}
bgColor_dict = {0 : 'yellow', 1 : 'cyan', 2 : 'tomato', 3 : 'lime', 4 : 'fuchsia', 5: 'orange', 6:'lightgray'}

def plotSpiral(spiralData):
    plt.figure()
    plt.scatter(*(extractElem(0, spiralData)).T, c=extractElem(1, spiralData), s=40)
    plt.axis('equal'); plt.savefig

def colorGetter(colorDict):
    return lambda colorKeys: [colorDict.get(colorKey, 'black') for colorKey in colorKeys]

markerColor, bgColor = [colorGetter(colorDict) for colorDict in [markerColor_dict, bgColor_dict]]

def lossPlotter(numEpochs, lossValues, numbClasses):
    f, ax = plt.subplots()
    ax.plot(range(1, 1+numEpochs), lossValues, '-', lw=3, c='k')
    ax.set_title(r'Cross entropy loss vs. epochs (log %d $\approx$ %.2f)' % (numbClasses, np.log(numbClasses)))
    ax.grid()
    plt.savefig(os.path.join('plotDir', str(numbClasses), "loss.png"))

def wrap_inputSpacePlotter(sess, positionData, dataHolder, labelData, outputLogits, numbClasses):
    saveDir = os.path.join('plotDir', str(numbClasses))
    def simplePlotter():
        testingGrid = makeGrid(positionData)
        gridResults = np.argmax(sess.run(outputLogits, {dataHolder: testingGrid}), 1)
        plt.figure(); plt.scatter(*testingGrid.T, c=bgColor(gridResults), s=10, alpha=0.3)
        plt.scatter(*positionData.T, c=markerColor(np.argmax(labelData, 1)), s=40, edgecolor='black', marker='s')
        plt.title('Decision boundaries @ input space ; %d classes' % numbClasses)
        plt.savefig('%s/inputData.png' % saveDir); plt.close()
    return simplePlotter

def wrap_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastHiddenLayer, outputLogits, numbClasses):
    def simplePlotter(epoch, backgroundClassFill):
        return utils_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastHiddenLayer, outputLogits, numbClasses, epoch, backgroundClassFill)
    return simplePlotter

def utils_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastHiddenLayer, outputLogits, numbClasses, epoch, backgroundClassFill):
    plt.figure()
    lastHiddenLayer_inputData = sess.run(lastHiddenLayer, {dataHolder: positionData})
    if backgroundClassFill:
        hiddenGrid = makeGrid(lastHiddenLayer_inputData)
        classProbabilities = sess.run(tf.nn.softmax(outputLogits), {lastHiddenLayer: hiddenGrid})
        plt.scatter(*hiddenGrid.T, c=bgColor(np.argmax(classProbabilities, 1)))
    plt.scatter(*lastHiddenLayer_inputData.T, s=40, edgecolor='black', marker='s', c=markerColor(np.argmax(labelData, 1)))
    titleDescription = lambda descriptiveString: plt.title(descriptiveString + ' @ last hidden layer ; %s ; %d classes' % (epoch, numbClasses))
    saveDirHelp = lambda descriptivePath: os.path.join(descriptivePath, str(numbClasses))
    if epoch == 'Final':
        titleDescription('Decision boundaries')
        saveDir = saveDirHelp('plotDir')
    else:
        titleDescription('Data representation')
        saveDir = saveDirHelp('framesDir')
    plt.savefig('%s/decisionBoundaries.%s.png' % (saveDir, epoch)); plt.close()

def wrap_vectorPlotter(sess, positionData, dataHolder, lastHiddenLayer, outputLogits, numbClasses):
    def simplePlotter():
        saveDir = os.path.join('plotDir', str(numbClasses))
        gridForClasses = makeGrid(positionData)
        gridClasses = np.argmax(sess.run(outputLogits, {dataHolder: gridForClasses}), 1)
        gridForArrows = makeGrid(positionData, 20)
        lastHiddenLayer_arrows = sess.run(lastHiddenLayer, {dataHolder: gridForArrows})
        plt.figure()
        plt.scatter(*gridForClasses.T, c=bgColor(gridClasses), alpha=0.3)
        plt.quiver(gridForArrows[:, 0], gridForArrows[:, 1], lastHiddenLayer_arrows[:, 0], lastHiddenLayer_arrows[:, 1], units='dots', headaxislength=2, headwidth=10, width=8)
        plt.title('Input space (2d) to last hidden layer (2d); vector plot'); plt.savefig('%s/vectorPlot.Guided.DataTransformer.png' % saveDir)
    return simplePlotter

def anglePlotter(angleDB, numbClasses):
    groupedAngleDB = angleDB.groupby('class')
    classesByIncreasingAngles = groupedAngleDB.apply(lambda group: (np.degrees(st.circmean(group['angle'])))).sort_values().index
    # assert len(set(angleDB['class'])) == numbClasses
    numbCols = 3 if numbClasses > 2 else 2
    numbRows = (numbClasses - 1) // numbCols + 1
    fig = plt.figure()
    for classID, classColor in enumerate(classesByIncreasingAngles):
        group = groupedAngleDB.get_group(classColor)
        meanAngle, angleSTD = np.degrees(st.circmean(group['angle'])), np.degrees(st.circstd(group['angle']))
        ax = fig.add_subplot(numbRows, numbCols, classID+1)
        ax.hist(np.degrees(group['angle']), 8, normed=True, histtype='bar', rwidth=0.8, color=classColor, edgecolor='k')
        ax.axvline(meanAngle, ls='--', c='k'); ax.set_yticks([])
        ax.set_title(r'$%.1f \pm %.1f$' % (meanAngle, angleSTD))
    fig.tight_layout(); plt.savefig(os.path.join('plotDir/', str(numbClasses), 'angles.png'))

def wrap_hiddenStatsBuilder(sess, positionData, dataHolder, outputLogits, lastHiddenLayer):
    def simpleBuilder():
        highResTestingGrid = makeGrid(positionData, 1000)
        gridResults = bgColor(np.argmax(sess.run(outputLogits, {dataHolder: highResTestingGrid}), 1))
        lastHiddenLayer_values = sess.run(lastHiddenLayer, {dataHolder: highResTestingGrid})
        angleDB = (np.arctan2(lastHiddenLayer_values[:, 1], lastHiddenLayer_values[:, 0]) + 2 * np.pi) % (2 * np.pi)
        return pd.DataFrame({'angle': angleDB, 'class': gridResults})
    return simpleBuilder
