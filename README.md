## Taking a peek into the hidden layers

With deep learning emerging a game-changer in virtually all areas of science, a question that keeps on appearing is: **"How & what do neural networks learn?"**

Despite a flurry of activity, the inner workings of these models work remain quite murky.  One interesting research direction is the so-called “manifold hypothesis” as mentioned by Chris Olah in an awesome [blog post](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).  According to this idea, *"the task of a classification algorithm is fundamentally to separate a bunch of tangled manifolds"* which does indeed sound very natural.

Let's explore this further by focusing on a very simple [synthetic dataset](http://cs231n.github.io/neural-networks-case-study/) of 2D interleaving spirals that belong to different classes.  The classification task can be achieved by a basic artificial neural network (MLP with 2 hidden layers).

The only "trick" is that the last hidden layer of the network has only 2 neurons.  Since the input is also in 2d, this means that we can visualize how the data flows from the input space to the last hidden in very straightforward way; it is a simple vector function from R2 to R2.  Because of the activation function, the hidden space takes bounded values in the square [-1, 1].

Let's first look at the process for 4 classes and you'll be able to see below how it is modified when changing the number of classes.

### A) Input space

The color of the points shows the class they belong to and the background color shows the class predicted by the trained neural net. 

<p align="center">
<img src="plotDir/4/inputData.png" width="420"/>
</p>

### B) Vector valued function from input to last hidden layer

<p align="center">
<img src="plotDir/4/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/4/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

### C) Training and final decision boundaries

What is really interesting is that classes try to divide this space by grouping each other 

<p align="center">
<img src="hidden.anim.4.gif" width="420"/>
<img src="plotDir/4/decisionBoundaries.Final.png" width="420"/>
</p>

### D) Optimal packing at the edges of a square?

Explain how to look at the packing problem in the square.

<p align="center">
<img src="plotDir/2/decisionBoundaries.Final.png" width="420"/>
<img src="plotDir/3/decisionBoundaries.Final.png" width="420"/>
<img src="plotDir/4/decisionBoundaries.Final.png" width="420"/>
<img src="plotDir/5/decisionBoundaries.Final.png" width="420"/>
<img src="plotDir/6/decisionBoundaries.Final.png" width="420"/>
<img src="plotDir/7/decisionBoundaries.Final.png" width="420"/>
</p>

### All the plots

##### 2 classes

<p align="center">
<img src="plotDir/2/inputData.png" width="420"/>
</p>

<p align="center">
<img src="plotDir/2/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/2/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

<p align="center">
<img src="hidden.anim.2.gif" width="420"/>
<img src="plotDir/2/decisionBoundaries.Final.png" width="420"/>
</p>

##### 3 classes

<p align="center">
<img src="plotDir/3/inputData.png" width="420"/>
</p>

<p align="center">
<img src="plotDir/3/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/3/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

<p align="center">
<img src="hidden.anim.3.gif" width="420"/>
<img src="plotDir/3/decisionBoundaries.Final.png" width="420"/>
</p>

##### 4 classes

<p align="center">
<img src="plotDir/4/inputData.png" width="420"/>
</p>

<p align="center">
<img src="plotDir/4/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/4/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

<p align="center">
<img src="hidden.anim.4.gif" width="420"/>
<img src="plotDir/4/decisionBoundaries.Final.png" width="420"/>
</p>

##### 5 classes

<p align="center">
<img src="plotDir/5/inputData.png" width="420"/>
</p>

<p align="center">
<img src="plotDir/5/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/5/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

<p align="center">
<img src="hidden.anim.5.gif" width="420"/>
<img src="plotDir/5/decisionBoundaries.Final.png" width="420"/>
</p>

##### 6 classes

<p align="center">
<img src="plotDir/6/inputData.png" width="420"/>
</p>

<p align="center">
<img src="plotDir/6/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/6/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

<p align="center">
<img src="hidden.anim.6.gif" width="420"/>
<img src="plotDir/6/decisionBoundaries.Final.png" width="420"/>
</p>

##### 7 classes

<p align="center">
<img src="plotDir/7/inputData.png" width="420"/>
</p>

<p align="center">
<img src="plotDir/7/vectorPlot.Raw.DataTransformer.png" width="420"/>
<img src="plotDir/7/vectorPlot.Guided.DataTransformer.png" width="420"/>
</p>

<p align="center">
<img src="hidden.anim.7.gif" width="420"/>
<img src="plotDir/7/decisionBoundaries.Final.png" width="420"/>
</p>
