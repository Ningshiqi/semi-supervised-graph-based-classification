from __future__ import division
import numpy as np
import tensorflow as tf
import argparse
import math
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
from scipy.sparse import csr_matrix
import json
import numpy as np
import random
import copy
from numpy import linalg as lin
from collections import defaultdict as dd
from scipy import sparse as sp
import argparse
import cPickle
import pickle
import networkx as nx
from sklearn.preprocessing import normalize
import os
from scipy.special import expit


#================================Settings================================
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 1, 'Initial learning rate.')
flags.DEFINE_float('embedding_size', 2, 'Size of the embedding vector.')
flags.DEFINE_float('num_sampled', 64, 'Number of negative samples sampled in the NCE.')
flags.DEFINE_integer('epochs', 5000002, 'Number of epochs to train.')
flags.DEFINE_string("data_dir", "/Users/danqinz/Desktop/semi-step/ligra/graph1/", "path to dataset")
flags.DEFINE_string("output_dir", "/Users/danqinz/Desktop/semi-step/graph/graph/graph1/", "path to output")
flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

#=================Helper Functions: Generate Context=================

def create_PPR_dictionary(filename):
    '''
    input: Ligra PPR vector txt file, direct output of C++ code
    output: PPR vector dictionary
        key: node ID (i)
        value: PPR vector for node i, in dictionary format:
            key: node ID (j)
            value: the probability of node j being sampled in a random walk starting from node i
    '''
    f = open(filename)
    count = 0
    names = []
    PPR = {}
    for line in f:
        if count%100 ==0:
            print count
        count += 1
        if line.split(':')[0] == '{"0"':
            PPR[0] = {}
            for item in line.split(':')[1].split('['):
                if item!='':
                    PPR[0][int(item.split(']')[0].split(',')[0])]= float(item.split(']')[0].split(',')[1])
        else:
            PPR[int(line.split(':')[0][1:][:-1])] = {}
            for item in line.split(':')[1].split('['):
                if item!='':
                    PPR[int(line.split(':')[0][1:][:-1])][int(item.split(']')[0].split(',')[0])]= float(item.split(']')[0].split(',')[1])
    return PPR
 
def parse_PPR_vector(PPR, context_size, num_sampling):
    """
    input: 
        PPR dictionary, direct output of create_PPR_dictionary function
        context_size: the number of nodes sampled as the context for a index node(starting node), similar to the length of random walk path
        num_sampling: # of sampling for each index node
    output: 
        sampled matrix: 
            # of rows = # of nodes * num_sampling
            # of columns = context_size +1, the first element is the index node, the rest are the context nodes
    """
    count = 0
    context = []
    for sampling in range(num_sampling):
        print 'sampling ',sampling
        context_matrix = np.zeros((len(PPR),context_size))
        for node in PPR:
            candidates = []
            prob = []
            for key in PPR[node]:
                candidates.append(int(key))
                prob.append(PPR[node][key])
            new_prob = np.array(prob)/sum(prob)
    #         print new_prob, candidates
            samples = np.random.choice(candidates, context_size-1, p=new_prob) 
            context_matrix[int(node),0] = node
            context_matrix[int(node),1:] = samples
        context.append(context_matrix)
    context_final = context[0]
    for i in range(1,len(context)):
        context_final = np.append(context_final,context[i], axis=0)
    return context_final
#=================Helper Functions: Similarity Check=================

def top_k(C,c_id,topk=5):
    '''
    input:
        C: embedding
        c_id: ID of the node you want to find similar nodes
        topk: top K nodes 
    output:
        a list of top K similar nodes ID, with the similarity score
    '''
    C_norm = normalize(C)
    c_vec=C_norm[c_id]
    sim = np.dot(C_norm,c_vec)
    nearest = (-sim).argsort()[1:topk+1]
    results=[(nearest[k],sim[nearest[k]]) for k in xrange(topk)]
    return results

def top_k_vec(C,vec,topk=5):
    '''
    input:
        C: embedding
        vec: the embedding vector of a node
        topk: top K nodes 
    output:
        a list of top K nodes ID whose embedding vectors are similar to the input embedding vector    
    '''
    C_norm=normalize(C)
    vec_norm=vec/np.linalg.norm(vec)
    sim = np.dot(C_norm,vec_norm)
    nearest = (-sim).argsort()[0:topk]
    return nearest

def print_top(results):
    '''
    input: output of the top_k function
    no output
    print the top_k function results node by node
    '''
    for result1,result2 in results:
        print result1,result2

def accuracy(predicted_y, y_real):
    '''
    calculate the prediction accuracy using predicted probability matrix and true label
    input:
        predicted_y: predicted probability matrix, N*2 matrix
        y_real: label vector, N*1 vector
    output:
        prediction accruacy
    '''
     count = 0
     for i in range(predicted_y.shape[0]):
         predict_label = np.argmax(predicted_y[i])
         if predict_label ==1 and y_real[i,1] ==1:
             count += 1
         elif predict_label ==0 and y_real[i,0]==1:
             count += 1
     return count/predicted_y.shape[0]
#================================Load Data================================

data_dir = FLAGS.data_dir
output_dir = FLAGS.output_dir
logs_dir = FLAGS.output_dir


# check if the context matrix exists in the data folder path
# if exists, load the data
# if not, generate the context matrix, and save the matrix
if os.path.isfile(data_dir + "new_graph1_context_5_10.npz")== True:
    random_walk_files="graph3_context_5.npz"
    np_random_walks=np.load(random_walk_files)['arr_0']
    np.random.shuffle(np_random_walks)
else:
    PPR = create_PPR_dictionary('../newgraph15.txt')
    context_matrix = parse_PPR_vector(PPR, 10, 100)
    np_random_walks=np.array(context_matrix,dtype=np.int32)
    np.savez(data_dir +"new_graph1_context_5_10.npz",np_random_walks)
    np.random.shuffle(np_random_walks)


# Load Data
# H: graph with N nodes in NetworkX format
# train_mask: N*1 boolean vector
# test_mask: N*1 boolean vector
# y_train: N*2 binary matrix
# y_true: N*2 binary matrix
# y_test: N*2 binary matrix
# X: feature matrix, N*d matrix

# Load the Graph in networkx format
H = pickle.load( open( data_dir + "H.p", "rb" ) )
print(H.number_of_nodes())
print(num_random_walks)
num_nodes= H.number_of_nodes()
num_random_walks = np_random_walks.shape[0]
# set batch size
batch_size = np_random_walks.shape[1]*(np_random_walks.shape[1]-1)

# Load the train and test mask
train_mask = pickle.load( open( data_dir + "train_mask.p", "rb" ) )
test_mask = pickle.load( open( data_dir + "test_mask.p", "rb" ) )

# generate the train index list and test index list based on train and test mask
train_index = []
test_index = []
for i in range(len(train_mask)):
    if train_mask[i]==True:
        train_index.append(i)
        
for i in range(len(test_mask)):
    if test_mask[i]==True:
        test_index.append(i)

# Load the labels
y_train = pickle.load( open( data_dir + "y_train.p", "rb" ) )
y_true = pickle.load( open( data_dir + "y_true.p", "rb" ) )
y_test = pickle.load( open( data_dir + "y_test.p", "rb" ) )


# Y_train: N*1 label vector
# Y_true: N*1 label vector
# Generate Y_train, Y_true based on the label files
Y_train = np.zeros(H.number_of_nodes())
for i in range(len(Y_train)):
    if y_train[i,0]==1:
        Y_train[i] = 1
    if y_train[i,1] ==1:
        Y_train[i]=-1
Y_true = np.zeros(H.number_of_nodes())
for i in range(len(Y_true)):
    if y_true[i,0]==1:
        Y_true[i] = 1
    if y_true[i,1] ==1:
        Y_true[i]=-1

# Load feature matrix      
X = pickle.load( open( data_dir + "X.p", "rb" ) )


#================================Set up Model================================
def generate_data_batch(np_random_walks, num_random_walks):
    '''
    generate data batch for deep learning training
    input:
        np_random_walks: numpy matrix format data of the random walks
        num_random_walks: total number of random walk paths
    output: pair of numpy array
        batch: an array of index nodes ID
        label: an array of context nodes ID
    '''
    dict_mapping = {}
    count = 0
    for i in range(num_random_walks):
        random_walk = np_random_walks[i]
        if random_walk[0] not in dict_mapping:
            dict_mapping[random_walk[0]] = [count]
        else:
            dict_mapping[random_walk[0]].append(count)
        count += 1
        i = np.random.choice(np_random_walks.shape[0])
        a_random_walk=np_random_walks[i]
        batch = []
        label = []

    for i in range(1,len(a_random_walk)):
        batch.append(a_random_walk[0])
        label.append(a_random_walk[i])

    for i in range(1,len(a_random_walk)):
        selected = np.random.choice(dict_mapping[a_random_walk[i]])
        b_random_walk = np_random_walks[selected]
        for i in range(1,len(b_random_walk)):
            batch.append(b_random_walk[0])
            label.append(b_random_walk[i])
    batch = np.array(batch)
    label = np.array(label).reshape(-1,1)
    return batch, label


graph = tf.Graph()
with graph.as_default():
    # initiate tf variables
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_logistics_ids = tf.placeholder(tf.int32, shape=[None,],name="input_node")
    train_y = tf.placeholder(tf.float32, [None, 2])
    train_x = tf.placeholder(tf.float32, [None, 2])
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    W = tf.Variable(tf.random_normal([embedding_size+X.shape[1], 2],stddev=0.35))
    b = tf.Variable(tf.zeros([2]))
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # unsupervised node embedding part
        embeddings = tf.Variable(
            tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([num_nodes, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
        loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=num_nodes))
        # supervised logistics regression part
        logistics_embed = tf.nn.embedding_lookup(embeddings, train_logistics_ids)
        logistics_x = tf.nn.embedding_lookup(train_x, train_logistics_ids)
        logistics_y = tf.nn.embedding_lookup(train_y, train_logistics_ids)
        logistics_input = tf.concat([logistics_embed, logistics_x], 1)
        pred = tf.nn.softmax(tf.matmul(logistics_input, W) + b) # Softmax
        loss_s = tf.reduce_mean(-tf.reduce_sum(logistics_y*tf.log(pred), reduction_indices=1))
        # add the unsupervised loss and supervised loss together
        loss = loss + 0.1*loss_s
        # Construct the SGD optimizer using a learning rate of defined in the setting part
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
        # normalize the embedding
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

#================================Start Training Session================================

#start session
if FLAGS.gpu_memory_fraction is not None:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    sess = tf.Session(config=config)
else:
    sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())
print("Initialized")
losses = []
for step in xrange(FLAGS.epochs):
    # generate the data batch for the epoch
    batch_inputs, batch_labels = generate_data_batch(np_random_walks,dict_mapping )
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, 
             train_y:y_train, train_logistics_ids:train_index}
    _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
    losses.append(loss_val)

    if step%50000 == 0:
        # get the normalized embedding vector for each node
        embed = sess.run(normalized_embeddings)
        # concatenate the embedding matrix with the feature matrix
        embed = np.concatenate((embed, X),axis=1)
        # calculate the probability distribution matrix
        predicted_y = np.ones((X.shape[0],2))
        predicted_y[:,0] = 1-expit(np.dot(embed,W)+b)
        predicted_y[:,1] = expit(np.dot(embed,W)+b)
        name = data_dir + "predicted_y_x_unsuper_"+str(step)+".p"
        pickle.dump( predicted_y, open( name, "wb" ) )  
