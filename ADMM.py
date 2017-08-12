from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.linalg as alg
import scipy as spy
import networkx as nx
import time
from itertools import *
import sys
import numpy.linalg as LA
import pickle
import random
import numpy as np
import scipy as spy
import time
from itertools import *
import sys
import cvxpy as cvx
from random import randint
import numpy as np
import random
from scipy.sparse import csc_matrix
from scipy import sparse as sp
import networkx as nx
from multiprocessing import Pool
import multiprocessing
from scipy.special import expit
from sklearn import linear_model, datasets
import argparse



class ADMM:
    '''
    ADMM for graph regularization Python class
    input: 
        X: feature matrix, N*d matrix
        y: N*1 label vector, where y_i = 0, if node i is in test indices
        temp: graph with N nodes in NetworkX format
        Lambda: hyperparameter to control graph regularization
        Rho: hyperparameter to control ADMM stepsize
        train_mask: N*1 boolean vector
        test_mask: N*1 boolean vector
        y_true: N*1 label vector
        Threshold:hyperparameter as the stopping criteria of ADMM algorithm
        paramters, features, labels, and graph structure
    output: 
        W: estimated W, d*1 vector
        b: estimated b, N*1 vector
        losses: losses per iteration
    '''
    def __init__(self, X, y, pos_node, temp, Lambda, Rho, train_mask, test_mask, y_true, Threshold):
        self.X = X
        self.y = y
        self.Threshold = Threshold
        self.y_true = y_true
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.dim = X.shape[1]
        self.Lambda = Lambda
        self.Rho = Rho
        self.temp = temp
        self.num_nodes = nx.number_of_nodes(self.temp)
        self.Z = csc_matrix((self.num_nodes, self.num_nodes), dtype=np.float).toarray()
        self.U = csc_matrix((self.num_nodes, self.num_nodes), dtype=np.float).toarray()
        for EI in self.temp.edges_iter():
            self.Z[EI[0],EI[1]] = np.random.rand()
            self.U[EI[0],EI[1]] = np.random.rand()
        logreg = linear_model.LogisticRegression()
        logreg.fit(X[self.train_mask], y[self.train_mask])
        # set the initial value of W and b with the logistics regression result
        self.W = logreg.coef_.reshape(X.shape[1])
        self.b = np.ones(X.shape[0])*logreg.intercept_
        self.pos_node = pos_node
        self.g = np.random.random((self.num_nodes,self.dim))
        self.h = np.random.random((self.num_nodes,self.dim))

    def update_W(self):
        loss = 0
        self.W = 0
        for i in range(self.num_nodes):
            self.W += (self.g[i,:] - self.h[i,:])/self.num_nodes

    def update_b(self):
        '''
        update the value of b, check line 4 of the ADMM algorithm for the math
        cvxpy is conducted independently for each node
        '''
        B = []
        for i in range(self.num_nodes):
            if i%50==0:
                print(i)
            bi = cvx.Variable(1)
            loss = cvx.logistic(-cvx.mul_elemwise(self.y[i], self.X[i].dot(self.g[i,:])+bi))*self.temp.node[i]['pos_node_prob']
            for Id in self.temp.neighbors(i):
                loss = loss+(bi-self.Z[i,Id]+ self.U[i,Id])**2*self.Rho/2
            problem = cvx.Problem(cvx.Minimize(loss))
            problem.solve(solver='SCS',verbose=False)
            self.b[i] = bi.value

    def update_g(self):
        '''
        update the value of g, check line 5 of the ADMM algorithm for the math
        cvxpy is conducted independently for each node
        this function is called in the runADMM_Grid() function
        '''
        for i in range(self.num_nodes):
            if i%50==0:
                print(i)
            gt = cvx.Variable(self.dim)
            loss = cvx.logistic(-cvx.mul_elemwise(self.y[i], self.X[i]*gt+self.b[i]))*self.temp.node[i]['pos_node_prob']
            loss += cvx.norm(self.W - gt + self.h[i,:])**2*self.Rho/2
            problem = cvx.Problem(cvx.Minimize(loss))
            problem.solve(solver='SCS',verbose=False)
            # if the optimization problem is not solved, then the result value of gt is None
            # if so, 
            if gt.value !=None:
                self.g[i,:] = gt.value.ravel()
            else:
                print('opt not solved')



    def update_Z(self):
        '''
        update the value of Z, check line 6 of the ADMM algorithm for the math
        '''
        for k in self.temp.nodes_iter():
            for j in self.temp.neighbors(k):
                A = self.b[j] + self.U[j,k]
                B = self.b[k] + self.U[k,j]
                self.Z[k,j] = (2*self.Lambda*self.temp[j][k]['pos_edge_prob']*A + (2*self.Lambda*self.temp[j][k]['pos_edge_prob']+self.Rho)*B)/(self.Lambda*4*self.temp[j][k]['pos_edge_prob']+self.Rho)

    def update_U(self):
        '''
        update the value of U, check line 7 of the ADMM algorithm for the math
        '''
        for i in self.temp.nodes_iter():
            for Id in self.temp.neighbors(i):
                self.U[i,Id] = self.U[i,Id] + self.b[i] - self.Z[i,Id]


    def update_h(self):
        '''
        update the valye of h, check line 7 of the ADMM algorithm for the math
        '''
        for i in range(self.num_nodes):
            self.h[i,:] = self.h[i,:] + (self.W -self.g[i,:])



    def runADMM_Grid(self):
        '''
        runADMM Grid iterations
        The stopping criteria is when the difference of the value of the objective function in current iteration
        and the value of the objective function in the previous iteration is smaller than the Threshold
        '''
        self.losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.times = []
        loss = self.cal_LL()
        train_accuracy = self.predict_train()
        self.losses.append(loss)
        self.train_accuracies.append(train_accuracy)
        test_accuracy = self.predict_test()
        self.test_accuracies.append(test_accuracy)
        print('iteration = ', 0)
        print('objective = ',loss)
        print(train_accuracy)
        print(test_accuracy)
        old_loss = 100000
        iterations = 0
        import time
        start = time.time()
        while np.absolute(old_loss- loss) >=  self.Threshold:
            old_loss = loss
            W_old = self.W
            b_old = self.b
            self.update_W()
            print('finished w')
            self.update_b()
            print('finished b')
            self.update_Z()
            print('finished Z')
            self.update_g()
            print('finished g')
            self.update_h()
            print('finished h')
            self.update_U()
            print('finished U')
            iterations += 1
            loss = self.cal_LL()
            print(iterations,loss)
            end = time.time()
            print('iteration',iterations)
            self.times.append(end-start)
            loss = self.cal_LL()
            train_accuracy = self.predict_train()
            self.losses.append(loss)
            self.train_accuracies.append(train_accuracy)
            test_accuracy = self.predict_test()
            self.test_accuracies.append(test_accuracy)
            print('iteration = ', iterations)
            print('objective = ', loss)
            print(train_accuracy)
            print(test_accuracy)
        print('total iterations = ',iterations)

    def cal_LL(self):
        '''
        function to calculate the value of loss function
        '''
        W = np.array(self.W).flatten()
        b = np.array(self.b).flatten()
        loss = np.sum(np.multiply(np.array(self.pos_node),np.log( (1+np.exp(-np.multiply(self.y,np.dot(self.X,W)+b))))))
        for EI in self.temp.edges_iter():
            loss +=  self.Lambda*(self.b[EI[0]]-self.b[EI[1]])**2*self.temp[EI[0]][EI[1]]['pos_edge_prob']
        return loss
    def predict_train(self):
        '''
        function to calculate the training accuracy
        '''
        predict = np.zeros((self.X[self.train_mask].shape[0],2))
        predict[:,0] = 1-expit(np.dot(self.X[self.train_mask],self.W)+self.b[self.train_mask])
        predict[:,1] = expit(np.dot(self.X[self.train_mask],self.W)+self.b[self.train_mask])
        predicted_y = np.argmax(predict,axis=1)
        return predict_accuracy(self.y[self.train_mask],predicted_y)
    def predict_test(self):
        '''
        function to calculate the testing accuracy
        '''
        predict = np.zeros((self.X[self.test_mask].shape[0],2))
        predict[:,0] = 1-expit(np.dot(self.X[self.test_mask],self.W)+self.b[self.test_mask])
        predict[:,1] = expit(np.dot(self.X[self.test_mask],self.W)+self.b[self.test_mask])
        predicted_y = np.argmax(predict,axis=1)
        return predict_accuracy(self.y_true[self.test_mask],predicted_y)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Lambda', type=float, default=0.1,
        help='hyperparameter to control graph regularization')
    parser.add_argument('--Rho', type=float, default=1.0,
        help='hyperparameter to control ADMM stepsize')
    parser.add_argument('--Threshold', type=float, default=5.0,
        help='hyperparameter as the stopping criteria of ADMM algorithm')
    parser.add_argument('--ADMM_output_file', type=str, default="C9.p",
        help='Name of output file of ADMM class')
    parser.add_argument('--plot_file', type=str, default="run9_losses_plot.png",
        help='Name of losses versus iteration plot')
    args = parser.parse_args()


    # Load the graph data in networkx graph format
    G3 = pickle.load(open("/home/danqinz/new_graph/graph1/G.p"))
    for u,v in G3.edges():
        G3[u][v]['pos_edge_prob'] = 1
    for i in range(G3.number_of_nodes()):
        G3.node[i]['pos_node_prob'] = 1
    # get all the nodes of the graph
    nodes = G3.nodes()
    # get some statistics about the graph
    print('number of nodes',G3.number_of_nodes())
    print('number of edges',G3.number_of_edges())

    # Load labels and mask for training and testing
    # y_train, y_true, y_test are N*2 binary matrix
    # train_mask, test_mask are N*1 boolean vector
    y_train = pickle.load( open( "/home/danqinz/new_graph/graph1/y_train.p", "rb" ) )
    y_true = pickle.load( open( "/home/danqinz/new_graph/graph1/Y_true.p", "rb" ) )
    y_test = pickle.load( open( "/home/danqinz/new_graph/graph1/y_test.p", "rb" ) )
    train_mask = pickle.load( open( "/home/danqinz/new_graph/graph1/train_mask.p", "rb" ) )
    test_mask = pickle.load( open( "/home/danqinz/new_graph/graph1/test_mask.p", "rb" ) )

    # convert y_train, y_true is get Y_train, Y_true
    # Y_train, Y_true are N*1 binary vector
    Y_train = np.zeros(G3.number_of_nodes())
    for i in range(len(Y_train)):
        if y_train[i,0]==1:
            Y_train[i] = -1
        if y_train[i,1] ==1:
            Y_train[i]=1
    Y_true = np.zeros(G3.number_of_nodes())
    for i in range(len(Y_true)):
        if y_true[i,0]==1:
            Y_true[i] = -1
        if y_true[i,1] ==1:
            Y_true[i]=1

    # Load feature matrix, select two features for the ADMM training
    X = pickle.load( open( "/home/danqinz/new_graph/graph1/X.p", "rb" ) )
    print(X.shape)
    X = X[:,[2,116]]

    def predict_accuracy(y,predicted_y):
        '''
        function to get prediction accuracy based on prediction result and true label
        '''
        count = 0
        for i in range(len(y)):
            if y[i]==-1 and predicted_y[i]==0:
                count += 1
            if y[i]==1 and predicted_y[i]==1:
                count += 1
        return count/len(y)

    # run the algorithm and time the code
    import time
    pos_node = np.ones(G3.number_of_nodes())
    C = ADMM(X, Y_train,pos_node,G3,args.Lambda, args.Rho, train_mask,test_mask,Y_true, args.Threshold)
    start = time.time()
    C.runADMM_Grid()
    end = time.time()
    print(end- start)

    # export the ADMM class to a pickle file
    pickle.dump( C, open( args.ADMM_output_file, "wb" ) )
    # export the losses versus iterations plot
    fig = plt.figure(3,figsize=(10,6))
    ax = plt.subplot(111)
    ax.plot(range(len(C.losses)),C.losses,label='train')
    plt.title('Training losses',fontsize=25)
    plt.legend(fontsize =15)
    fig.savefig(args.plot_file)
    plt.close()
