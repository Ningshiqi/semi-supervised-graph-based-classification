from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
import pickle
import time
import tensorflow as tf
import scipy
from gcn.utils import *
from gcn.models import GCN, MLP
import matplotlib.pyplot as plt
import networkx as nx

#================================Settings================================

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

'''
# Settings
    model can be changed from gcn_cheby to gcn, they are two different models
    gcn_cheby model is from this paper: https://arxiv.org/abs/1606.09375
    gcn model is from this this paper: https://arxiv.org/abs/1609.02907
'''
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string("data_dir", "/Users/danqinz/Desktop/semi-step/ligra/graph1/", "path to dataset")
flags.DEFINE_string("output_dir", "/Users/danqinz/Desktop/semi-step/graph/graph/graph1/", "path to output")
flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

#================================Load Data================================

# Load Data
# G3: graph with N nodes in NetworkX format
# adj: adj matrix of G3 in scipy sparse matrix format
# Y_true: N*1 label vector
# train_mask: N*1 boolean vector
# test_mask: N*1 boolean vector
# X: feature matrix, N*d matrix


data_dir = FLAGS.data_dir
output_dir = FLAGS.output_dir
logs_dir = FLAGS.output_dir
G3 = pickle.load(open(data_dir+"G.p"))
adj = pickle.load(open(data_dir+"adj.p"))
nodes = G3.nodes()
print('number of nodes',G3.number_of_nodes())
print('number of edges',G3.number_of_edges())
Y_true = pickle.load(open(data_dir + "Y_train.p"))
train_mask = pickle.load( open( data_dir + "train_mask.p", "rb" ) )
test_mask = pickle.load( open( data_dir + "test_mask.p", "rb" ) )
y_true = np.zeros((G3.number_of_nodes(),2))
y_train = np.zeros((G3.number_of_nodes(),2))
y_test = np.zeros((G3.number_of_nodes(),2))
for i in G3.nodes():
    if Y_true[i]==1:
        y_true[i,1] = 1
        if train_mask[i]==True:
            y_train[i,1] =1
        else:
            y_test[i,1]=1
    else:
        y_true[i,0] = 1
        if train_mask[i]==True:
            y_train[i,0] =1
        else:
            y_test[i,0]=1 
X = pickle.load( open( data_dir + "X.p", "rb" ) )
features = scipy.sparse.lil_matrix(X)
print(features.shape)
y_val = np.copy(y_train)   
val_mask = np.copy(train_mask)

#================================Set up Model================================
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

#set up summaries
#training loss summaries
train_loss_summary = []
train_loss_summary.append(tf.summary.scalar("training_loss", model.loss))
train_loss_summary.append(tf.summary.scalar("training_accuracy", model.accuracy))
train_loss_summary = tf.summary.merge(train_loss_summary)

#================================Start Training Session================================
#start session
if FLAGS.gpu_memory_fraction is not None:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    sess = tf.Session(config=config)
else:
    sess = tf.Session()

#set up savers
saver = tf.train.Saver(max_to_keep=20)
summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
cost_val = []



# Train model
test_accuracies = []
train_accuracies = []
for epoch in range(201):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    ##### Training step
    # outputs are:
    # outs[1]: loss 
    # outs[2]: accuracy(can be training/testing/validation with different feed_dict)
    # outs[3]: probability distribution matrix, N*2 matrix, used for ROC curve plot and model comparison
    outs = sess.run([model.opt_op, model.loss, model.accuracy, tf.nn.softmax(model.outputs)], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Cost:",cost)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    train_accuracies.append(acc)
    # test results
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "test accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    test_accuracies.append(test_acc)
    # save summary in log directory for tensorboard visualization
    summary_str = sess.run(train_loss_summary, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, itr)
    # save the probability distribution matrix every 10 iterations for ROC curve olit and model comparision
    if epoch%10==0 and epoch!=0:
        name = output_dir + "GCN_" + str(epoch) + '.p'
        pickle.dump( outs[3], open( name, "wb" ) )    