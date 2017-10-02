# semi-supervised-graph-based-classification
## Overview
Three directions of semi-supervised graph-based classification
* skip-gram node embedding
* graph regularization
* deep learning(graph convolutional networks)

We came up with new methods for direction 1 and direction 2. Code implemented in Python, method1 and method3 are implemented using tensorflow, and method2 we have implemented self-drived algorithm. Method3 is adapted from the [first author's implementation](https://github.com/tkipf/gcn) on github, you will have to call the Python class from the original repo to run the GCN.py code. All codes tested on real dataset. 

## Markov Random Field
### (1) MCMC and Mean Field/ Loopy Belief Propagation
The toy example of the naive mean field with closed form update solution is implemented in [this notebook](https://github.com/DanqingZ/semi-supervised-graph-based-classification/blob/master/mean_field.ipynb). 

### (2) Embedding Mean Field/ Loopy Belief Propagation
Implement structure2vec from [Hanjun Dai's paper](https://arxiv.org/abs/1603.05629) in Pure Python. The vanila version of the embedding mean field is implemented in [this notebook](https://github.com/DanqingZ/semi-supervised-graph-based-classification/blob/master/mean_field.ipynb).

### (3)Discriminative Mean Field
The toy version is implemented in [this notebook](https://github.com/DanqingZ/semi-supervised-graph-based-classification/blob/master/structure2vec_toy.ipynb). Check [graphsage](https://github.com/williamleif/GraphSAGE) mean aggregator method, which is very similar to the Discriminative Mean Field.

### (4) Exploration
Joint probability distribution of two nodes on an edge in the Markov Random Field. This problem is not of interest to the graph node classification, but is worth exploring.
