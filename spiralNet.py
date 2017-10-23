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

numbClasses = 4
numEpochs = 4000

learningRate, L2reg = 0.0005, 0.005

dataHolder = tf.placeholder(tf.float32, [None, 2]) # data is in R2
labelsHolder = tf.placeholder(tf.float32, [None, numbClasses])

hidden0 = tf.layers.dense(inputs=dataHolder, units=120, activation=tf.tanh, name='hidden0')
hidden1 = tf.layers.dense(inputs=hidden0, units=80, activation=tf.tanh, name='hidden1')
hidden11 = tf.layers.dense(inputs=hidden1, units=80, activation=tf.tanh, name='hidden11')

lastHiddenLayer = tf.layers.dense(inputs=hidden11, units=2, activation=tf.tanh, name='lastHiddenLayer')

outputLogits = tf.layers.dense(inputs=lastHiddenLayer, units=numbClasses, name='outputLogits')

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
    sess.run(init)

    positionData, labelData = extractElem(0, spiralData), tf.one_hot(extractElem(1, spiralData), depth=numbClasses).eval()

    # simple closures to simplify calls to plotting functions
    inputSpace_plotter = wrap_inputSpacePlotter(sess, positionData, dataHolder, labelData, outputLogits, numbClasses)
    hiddenLayer_plotter = wrap_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastHiddenLayer, outputLogits, numbClasses)
    inputToHidden_vectorPlotter = wrap_vectorPlotter(sess, positionData, dataHolder, lastHiddenLayer, outputLogits, numbClasses)

    crossEntropyLossValues = []

    for epoch in range(numEpochs):
        _, _lossValue, _crossEntropyLoss, _L2RegLoss = sess.run([optimizer, lossValue, crossEntropyLoss, L2RegLoss], {dataHolder: positionData, labelsHolder: labelData})
        crossEntropyLossValues.append(_crossEntropyLoss)
        _accuracy = accuracy.eval({dataHolder: positionData, labelsHolder: labelData})
        hiddenLayer_plotter(epoch, backgroundClassFill=False) if epoch in hiddenCheckPoints else None
        print('%d\t\tcrossEntropy = %f\tL2 = %f\ttotal = %f\taccuracy = %f' % (epoch, _crossEntropyLoss, _L2RegLoss, _lossValue, _accuracy))

    print('Optimization Finished!')

    lossPlotter(numEpochs, crossEntropyLossValues, os.path.join('plotDir/', str(numbClasses)))
    inputSpace_plotter()
    hiddenLayer_plotter('Final', backgroundClassFill=True)
    inputToHidden_vectorPlotter()

    highResTestingGrid = makeGrid(positionData, 1000)
    gridResults = bgColor(np.argmax(sess.run(outputLogits, {dataHolder: highResTestingGrid}), 1))
    lastHiddenLayer_values = lastHiddenLayer.eval({dataHolder: highResTestingGrid}) 
    allAngles = (np.arctan2(lastHiddenLayer_values[:, 1], lastHiddenLayer_values[:, 0]) + 2 * np.pi) % (2 * np.pi)
    angleDB = pd.DataFrame({'angle': allAngles, 'class': gridResults})

    # TODO: Emphazise that we are using the circular mean in order to deal with the wraparound

    statData = angleDB.groupby('class').apply(lambda gr: pd.Series({'meanAngle': st.circmean(gr['angle']) * 180 / np.pi, 'angleSTD': st.circstd(gr['angle']) * 180 / np.pi})).sort_values('meanAngle')
    statData['diff'] = statData['meanAngle'] - statData['meanAngle'].shift(1)

    print(statData)

    #import ipdb; ipdb.set_trace(context=30)

    # optimizedWeights = {x.name: tf.get_default_graph().get_tensor_by_name(x.name).eval() for x in tf.trainable_variables()}

