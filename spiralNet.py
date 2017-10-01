#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Utils.spiral import makeSpiral
from Utils.plottingUtils import *
from Utils.generalUtils import makeGrid

numbClasses = 5
plotDir = 'plotDir'
learningRate = 0.0005
L2reg = 0.005
numEpochs = 8000

dataHolder = tf.placeholder(tf.float32, [None, 2]) # data is in R2
labelsHolder = tf.placeholder(tf.float32, [None, numbClasses])

hidden1 = tf.layers.dense(inputs=dataHolder, units=40, activation=tf.tanh, name='hidden1')
hidden2 = tf.layers.dense(inputs=hidden1, units=2, activation=tf.tanh, name='hidden2')
outputLogits = tf.layers.dense(inputs=hidden2, units=numbClasses, name='outputLogits')

crossEntropyLoss = tf.losses.softmax_cross_entropy(logits=outputLogits, onehot_labels=labelsHolder)
L2RegLoss = L2reg * tf.add_n([tf.nn.l2_loss(layerParameters) for layerParameters in tf.trainable_variables() if 'bias' not in layerParameters.name])
lossValue = crossEntropyLoss + L2RegLoss

optimizer = tf.train.AdamOptimizer(learningRate).minimize(lossValue)

correctPredictions = tf.equal(tf.argmax(outputLogits, 1), tf.argmax(labelsHolder, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

init = tf.global_variables_initializer()

spiralData = list(map(makeSpiral, range(numbClasses)))

with tf.Session() as sess:
    sess.run(init)
    positionData, labelData = extractElem(0, spiralData), tf.one_hot(extractElem(1, spiralData), depth=numbClasses).eval(session=sess)

    crossEntropyLossValues = []

    for epoch in range(numEpochs):
        _, _lossValue, _crossEntropyLoss, _L2RegLoss = sess.run([optimizer, lossValue, crossEntropyLoss, L2RegLoss], {dataHolder: positionData, labelsHolder: labelData})
        crossEntropyLossValues.append(_crossEntropyLoss)
        _accuracy = accuracy.eval({dataHolder: positionData, labelsHolder: labelData})
        print('%d\t\tcrossEntropy = %f\tL2 = %f\ttotal = %f\taccuracy = %f' % (epoch, _crossEntropyLoss, _L2RegLoss, _lossValue, _accuracy))

    print('Optimization Finished!')
    lossPlotter(numEpochs, crossEntropyLossValues, plotDir)
    
    testingGrid = makeGrid(positionData)
    gridResults = np.argmax(sess.run(outputLogits, {dataHolder: testingGrid}), 1)
    plotData(testingGrid, gridResults, positionData, labelData, plotDir)

    optimizedWeights = {x.name: tf.get_default_graph().get_tensor_by_name(x.name).eval(session=sess) for x in tf.trainable_variables()}

    hidden2Values = hidden2.eval({dataHolder: positionData})
    hiddenGrid = makeGrid(hidden2Values)
    probMap = tf.nn.softmax(np.dot(hiddenGrid, optimizedWeights['outputLogits/kernel:0']) + optimizedWeights['outputLogits/bias:0']).eval()
    f, ax = plt.subplots()
    ax.scatter(hiddenGrid[:, 0], hiddenGrid[:, 1], c=bgColor(np.argmax(probMap, 1)))
    ax.scatter(hidden2Values[:, 0], hidden2Values[:, 1], s=40, cmap=plt.cm.Spectral, edgecolor='black', marker='s', linewidth=1, c=markerColor(np.argmax(labelData, 1)))
    ax.set_title('Decision boundaries @ last hidden layer')
    plt.savefig('%s/decisionBoundaries.png' % plotDir); plt.close()

