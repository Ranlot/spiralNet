#!/usr/bin/env python3

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Utils.spiral import makeSpiral
from Utils.plottingUtils import *

parser = argparse.ArgumentParser()
parser.add_argument('--numbClasses', type=int, required=True, dest='numbClasses', help='number of classes')
parser.add_argument('--numbEpochs', type=int, required=True, dest='numbEpochs', help='number of epochs')
parser.add_argument('--activation', type=str, required=True, dest='activation', help='activation function ; only tanh and ReLU are supported')
args = vars(parser.parse_args())

activDir = {'tanh': tf.tanh, 'ReLU': tf.nn.relu}
numbClasses, numEpochs, activation = args['numbClasses'], args['numbEpochs'], activDir[args['activation']]
learningRate, L2reg = 0.0005, 0.005

dataHolder = tf.placeholder(tf.float32, [None, 2]) # data is in R2
labelsHolder = tf.placeholder(tf.float32, [None, numbClasses])

hidden0 = tf.layers.dense(inputs=dataHolder, units=120, activation=activation, name='hidden0')
hidden1 = tf.layers.dense(inputs=hidden0, units=80, activation=activation, name='hidden1')
hidden2 = tf.layers.dense(inputs=hidden1, units=40, activation=activation, name='hidden2')

lastHiddenLayer = tf.layers.dense(inputs=hidden2, units=2, activation=activation, name='lastHiddenLayer')

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

with tf.Session() as sess:
    sess.run(init)

    positionData, labelData = extractElem(0, spiralData), tf.one_hot(extractElem(1, spiralData), depth=numbClasses).eval()

    # simple closures to simplify calls to plotting functions and statitistics builder
    inputSpace_plotter = wrap_inputSpacePlotter(sess, positionData, dataHolder, labelData, outputLogits, numbClasses, args['activation'])
    hiddenLayer_plotter = wrap_hiddenLayerPlotter(sess, positionData, dataHolder, labelData, lastHiddenLayer, outputLogits, numbClasses, args['activation'])
    inputToHidden_vectorPlotter = wrap_vectorPlotter(sess, positionData, dataHolder, lastHiddenLayer, outputLogits, numbClasses, args['activation'])
    hiddenStats_builder = wrap_hiddenStatsBuilder(sess, positionData, dataHolder, outputLogits, lastHiddenLayer)

    crossEntropyLossValues = []
    for epoch in range(numEpochs):
        _, _lossValue, _crossEntropyLoss, _L2RegLoss = sess.run([optimizer, lossValue, crossEntropyLoss, L2RegLoss], {dataHolder: positionData, labelsHolder: labelData})
        crossEntropyLossValues.append(_crossEntropyLoss)
        _accuracy = accuracy.eval({dataHolder: positionData, labelsHolder: labelData})
        hiddenLayer_plotter(epoch, backgroundClassFill=False) if epoch in hiddenCheckPoints else None
        print('%d\t\tcrossEntropy = %f\tL2 = %f\ttotal = %f\taccuracy = %f' % (epoch, _crossEntropyLoss, _L2RegLoss, _lossValue, _accuracy))
    print('\nOptimization Finished!\nPreparing plots...\n')

    lossPlotter(numEpochs, crossEntropyLossValues, numbClasses, args['activation'])
    inputSpace_plotter()
    hiddenLayer_plotter('Final', backgroundClassFill=True)
    inputToHidden_vectorPlotter()

    statsDB = hiddenStats_builder() # prepare some statistics about input to last hidden layer function
    anglePlotter(statsDB, numbClasses, args['activation'])
