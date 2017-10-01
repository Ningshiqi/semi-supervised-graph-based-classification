# semi-supervised-graph-based-classification
## Overview
Three directions of semi-supervised graph-based classification
* skip-gram node embedding
* graph regularization
* deep learning(graph convolutional networks)

We came up with new methods for direction 1 and direction 2. Code implemented in Python, method1 and method3 are implemented using tensorflow, and method2 we have implemented self-drived algorithm. Method3 is adapted from the [first author's implementation](https://github.com/tkipf/gcn) on github. All codes tested on real dataset. 

## New direction
Implement structure2vec from [Hanjun's paper](https://arxiv.org/abs/1603.05629) in Pure Python. The toy example of the naive mean field with closed form update solution is implemented in [this notebook](https://github.com/DanqingZ/semi-supervised-graph-based-classification/blob/master/mean_field.ipynb). The vanila version of the embedding mean field is implemented in [this notebook](https://github.com/DanqingZ/semi-supervised-graph-based-classification/blob/master/mean_field.ipynb).
