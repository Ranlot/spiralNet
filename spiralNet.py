#!/usr/bin/env python3

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.stats as st
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Utils.spiral import makeSpiral
from Utils.plottingUtils import *
from Utils.generalUtils import makeGrid

numbClasses = 2
plotDir = os.path.join('plotDir/', str(numbClasses))
learningRate = 0.0005
L2reg = 0.005
numEpochs = 5000

dataHolder = tf.placeholder(tf.float32, [None, 2]) # data is in R2
labelsHolder = tf.placeholder(tf.float32, [None, numbClasses])

hidden0 = tf.layers.dense(inputs=dataHolder, units=40, activation=tf.tanh, name='hidden0')
hidden1 = tf.layers.dense(inputs=hidden0, units=40, activation=tf.tanh, name='hidden1')
hidden2 = tf.layers.dense(inputs=hidden1, units=2, activation=tf.tanh, name='hidden2')
outputLogits = tf.layers.dense(inputs=hidden2, units=numbClasses, name='outputLogits')

crossEntropyLoss = tf.losses.softmax_cross_entropy(logits=outputLogits, onehot_labels=labelsHolder)
L2RegLoss = L2reg * tf.add_n([tf.nn.l2_loss(layerParameters) for layerParameters in tf.trainable_variables() if 'bias' not in layerParameters.name])
lossValue = crossEntropyLoss + L2RegLoss

optimizer = tf.train.AdamOptimizer(learningRate).minimize(lossValue)

correctPredictions = tf.equal(tf.argmax(outputLogits, 1), tf.argmax(labelsHolder, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

init = tf.global_variables_initializer()

spiralData = list(map(lambda x: makeSpiral(x, numbClasses), range(numbClasses)))

hiddenCheckPoints = set(map(int, np.logspace(np.log10(1), np.log10(numEpochs-1), 350)))

# x = [np.degrees(np.arctan2(_[1], _[0])) for _ in hidden2Values]
# hist, bins = np.histogram(x, bins=100)
# center = (bins[:-1] + bins[1:]) / 2
# plt.figure(); plt.scatter(center, hist); plt.savefig('tt2.png')
# probMap = bgColor(np.argmax(tf.nn.softmax(outputLogits).eval({dataHolder: testingGrid}), 1))
# [(z, np.array([np.degrees(np.arctan2(_[0][1], _[0][0])) for _ in zip(hidden2Values, probMap) if _[1] == z]).mean()) for z in set(probMap)]
# [(z, np.array([np.degrees(np.arctan2(_[0][1], _[0][0])) for _ in zip(hidden2Values, probMap) if _[1] == z]).std()) for z in set(probMap)]
# [360 / 5. * _ for _ in range(5)]

# the function is trying to pull points apart in maximum opposite directions
# TODO: define analytical functions that would achieve exactly this with cos / sin ; maybe a simple geometric function would exist

with tf.Session() as sess:

    def hiddenLayerPlotter(epoch='Final', gridFill=False, saveDir=os.path.join('framesDir', str(numbClasses))):
        f, ax = plt.subplots()
        hidden2Values = hidden2.eval({dataHolder: positionData})
        if gridFill:
            hiddenGrid = makeGrid(hidden2Values)
            probMap = tf.nn.softmax(outputLogits).eval({hidden2: hiddenGrid})
            ax.scatter(hiddenGrid[:, 0], hiddenGrid[:, 1], c=bgColor(np.argmax(probMap, 1)))
        ax.scatter(hidden2Values[:, 0], hidden2Values[:, 1], s=40, cmap=plt.cm.Spectral, edgecolor='black', marker='s', linewidth=1, c=markerColor(np.argmax(labelData, 1)))
        ax.set_title('Decision boundaries @ last hidden layer ; %s' % epoch)
        plt.savefig('%s/decisionBoundaries.%s.png' % (saveDir, epoch)); plt.close()
     
    sess.run(init)
    positionData, labelData = extractElem(0, spiralData), tf.one_hot(extractElem(1, spiralData), depth=numbClasses).eval()

    crossEntropyLossValues = []

    for epoch in range(numEpochs):
        _, _lossValue, _crossEntropyLoss, _L2RegLoss = sess.run([optimizer, lossValue, crossEntropyLoss, L2RegLoss], {dataHolder: positionData, labelsHolder: labelData})
        crossEntropyLossValues.append(_crossEntropyLoss)
        _accuracy = accuracy.eval({dataHolder: positionData, labelsHolder: labelData})
        if epoch in hiddenCheckPoints:	hiddenLayerPlotter(epoch)
        print('%d\t\tcrossEntropy = %f\tL2 = %f\ttotal = %f\taccuracy = %f' % (epoch, _crossEntropyLoss, _L2RegLoss, _lossValue, _accuracy))

    print('Optimization Finished!')
    lossPlotter(numEpochs, crossEntropyLossValues, plotDir)

    #import ipdb; ipdb.set_trace(context=30)

    testingGrid = makeGrid(positionData)
    gridResults = np.argmax(sess.run(outputLogits, {dataHolder: testingGrid}), 1)
    plotData(testingGrid, gridResults, positionData, labelData, plotDir)


    smallTestingGrid = makeGrid(positionData, 20)

    testingGrid = makeGrid(positionData)
    gridResults = np.argmax(sess.run(outputLogits, {dataHolder: testingGrid}), 1)
    
    hidden2Values = hidden2.eval({dataHolder: smallTestingGrid})
    probMap = np.argmax((outputLogits).eval({dataHolder: smallTestingGrid}), 1)
    # cmap = matplotlib.colors.ListedColormap(list(markerColor_dict.values()))
    plt.figure()
    plt.scatter(testingGrid[:, 0], testingGrid[:, 1], c=bgColor(gridResults), alpha=0.3)
    # plt.quiver(smallTestingGrid[:, 0], smallTestingGrid[:, 1], hidden2Values[:, 0], hidden2Values[:, 1], probMap, cmap=cmap, units='dots', headaxislength=2, headwidth=10, width=8)
    plt.quiver(smallTestingGrid[:, 0], smallTestingGrid[:, 1], hidden2Values[:, 0], hidden2Values[:, 1], units='dots', headaxislength=2, headwidth=10, width=8)
    plt.savefig('%s/vectorPlot.Guided.DataTransformer.png' % plotDir)

    plt.figure()
    plt.quiver(smallTestingGrid[:, 0], smallTestingGrid[:, 1], hidden2Values[:, 0], hidden2Values[:, 1], units='dots', headaxislength=2, headwidth=10, width=8)
    plt.savefig('%s/vectorPlot.Raw.DataTransformer.png' % plotDir)

    # import ipdb; ipdb.set_trace(context=30)

    highResTestingGrid = makeGrid(positionData, 1000)
    gridResults = bgColor(np.argmax(sess.run(outputLogits, {dataHolder: highResTestingGrid}), 1))
    hidden2Values = hidden2.eval({dataHolder: highResTestingGrid}) 
    allAngles = (np.arctan2(hidden2Values[:, 1], hidden2Values[:, 0]) + 2 * np.pi) % (2 * np.pi)
    angleDB = pd.DataFrame({'angle': allAngles, 'class': gridResults})

    # TODO: Emphazise that we are using the circular mean in order to deal with the wraparound

    statData = angleDB.groupby('class').apply(lambda gr: pd.Series({'meanAngle': st.circmean(gr['angle']) * 180 / np.pi, 'angleSTD': st.circstd(gr['angle']) * 180 / np.pi})).sort_values('meanAngle')
    statData['diff'] = statData['meanAngle'] - statData['meanAngle'].shift(1)

    print(statData)

    #import ipdb; ipdb.set_trace(context=30)

    # optimizedWeights = {x.name: tf.get_default_graph().get_tensor_by_name(x.name).eval() for x in tf.trainable_variables()}

    hiddenLayerPlotter(gridFill=True, saveDir=plotDir) 
