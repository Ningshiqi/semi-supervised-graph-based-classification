from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
# get_ipython().magic(u'matplotlib inline')
# # matplotlib.use('Agg')
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
# set hyperparameter Lambda and Rho
Lambda = 0.1
Rho = 1

import random
import numpy as np
import numpy.linalg as LA
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
import logging
import collections
import math
import csv
import pandas as pd
from multiprocessing import Pool
from functools import partial
import multiprocessing
import ctypes
import numpy as np
import time
# import models from other files
# from Model import *


shared_array_base = multiprocessing.Array(ctypes.c_double, 99993)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(99993,)



def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)
# In[221]:

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class ADMM_par:
    '''
    ADMM for graph regularization Python class
    input: 
        X: feature matrix, N*d matrix
        y: N*1 label vector, where y_i = 0, if node i is in test indices
        G: graph with N nodes as a nested dictionary
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
    def __init__(self, X, y, G, nodes, edgeNbr, Lambda, Rho, train_mask, test_mask, y_true, Threshold, initialW, initialb, Z, U):
        self.X = X
        self.y = y
        self.Threshold = Threshold
        self.y_true = y_true
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.dim = X.shape[1]
        self.Lambda = Lambda
        self.Rho = Rho
        self.graph = G
        self.nodes = nodes
        self.count_b = 0

        row=[]
        col=[]
        for i, js in self.graph.items():
            for j in js:
                row.append(i)
                col.append(j)
                
        initialZ = Z
        initialU = U
        self.Z = collections.defaultdict(dict)
        self.U = collections.defaultdict(dict)
        k=0
        for i, js in self.graph.items():
            for j in js:
                self.Z[i][j]=initialZ[k]
                self.U[i][j]=initialU[k]
                k+=1

        # set the initial value of W and b with the logistics regression result
        self.W = initialW.reshape(X.shape[1])
        shared_array[:] = initialb

    def dumpWb(self, filename):
        dict = {"W": self.W, "b": shared_array}
        with open(filename, "wb") as f:
            pickle.dump( dict, f)
        
    def deriv_b(self, b, C1, C2, C3, eC1):
        if (eC1 == float('inf')):
            return C2 + C3 *b
    
        return 1/(1+ eC1* math.exp(-1.0*b)) + C2 + C3*b

    def deriv_b_negy(self, b, C1, C2, C3, eC1):
        if (eC1 == float('inf')):
            return C2 + C3 *b
        
        return -1.0/(1+ eC1* math.exp(b)) + C2 + C3*b
#     def worker(self, b, start,send_end): 
#         send_end.send(b)
        
    def worker(self, start, end, def_param=shared_array):
        '''
        update the value of b, check line 4 of the ADMM algorithm for the math
        cvxpy is conducted independently for each node
        '''
        B = []
        kk=0
        for i in range(start, end):
            sumdiffZU = 0
            neighborCnt = 0
            for Id in self.graph[i]:
                sumdiffZU += (self.Z[i][Id]-self.U[i][Id])
                neighborCnt += 1

            if (neighborCnt == 0):
                 raise ValueError('{0} has no neighbor'.format(i))

            b1 = sumdiffZU /neighborCnt

            #in case of missing value, we have analytical solution for b
            if (self.y[i]==0):
                shared_array[i] = b1
                continue
                # b[j]= b1
                # self.count_b += 1
                # print(self.count_b)
                # continue

            tol = 1e-5

            #the optimial value is within the interval [b1, b2]
            if (self.y[i]==1):
                b2 = b1 + 1/self.Rho/neighborCnt
                #bisection method to find a better b
                C1 = -1.0 * self.X[i].dot(self.W) #C1 = -1.0 * self.X[i].dot(self.g[i,:])
                C2 = -1-self.Rho * sumdiffZU
                C3 = self.Rho * neighborCnt
                eC1 = 0
                try:
                    eC1 = math.exp(C1)
                except OverflowError:
                    eC1 = float('inf')
                while(b2-b1 > tol):
                    Db1 = self.deriv_b(b1, C1, C2, C3, eC1)
                    Db2 = self.deriv_b(b2, C1, C2, C3, eC1)
                    if (math.fabs(Db1)<tol):
                        b2 = b1
                        break;

                    if (math.fabs(Db2)<tol):
                        b1 = b2
                        break;
                    
                    if (not(Db1<=tol and Db2>=-1.0*tol)):
                        raise ValueError('Db1 and Db2 has same sign which is impossible! Db1={0}, Db2={1}, b1={2}, b2={3}'.format(Db1, Db2, b1, b2))
                    
                    b3 = (b1 + b2)/2
                    Db3 = self.deriv_b(b3, C1, C2, C3, eC1)
                    if (Db3 >=0):
                        b2=b3
                    else:
                        b1=b3
                    
                # b[j] = (b1 + b2)/2
                shared_array[i] = (b1 + b2)/2
                continue
                # print('finished the computation of b for node ',i)
                # self.count_b += 1
                # print(self.count_b)
                # continue

            if (self.y[i]==-1):
                b2 = b1
                b1 = b2 - 1/self.Rho/neighborCnt
                C1 = self.X[i].dot(self.W) #C1 = self.X[i].dot(self.g[i,:])
                C2 = 1-self.Rho * sumdiffZU
                C3 = self.Rho * neighborCnt
                eC1 = 0
                try:
                    eC1 = math.exp(C1)
                except OverflowError:
                    eC1 = float('inf')
                    
                while(b2-b1 > tol):
                    Db1 = self.deriv_b_negy(b1, C1, C2, C3, eC1)
                    Db2 = self.deriv_b_negy(b2, C1, C2, C3, eC1)
                    if (math.fabs(Db1)<tol):
                        b2 = b1
                        break;

                    if (math.fabs(Db2)<tol):
                        b1 = b2
                        break;
                    
                    if (not(Db1<=tol and Db2>=-1.0*tol)):
                        raise ValueError('Db1 and Db2 has same sign which is impossible! Db1={0}, Db2={1}, b1={2}, b2={3}, C1={4}, C2={5}, C3={6}'.format(
                            Db1, Db2, b1, b2, C1, C2, C3))
                    
                    b3 = (b1 + b2)/2
                    Db3 = self.deriv_b_negy(b3, C1, C2, C3, eC1)
                    if (Db3 >=0):
                        b2=b3
                    else:
                        b1=b3
                        
                    
                # b[j] = (b1 + b2)/2
                shared_array[i] = (b1+b2)/2
                # print('finished the computation of b for node ',i)
                # self.count_b += 1
                # print(self.count_b)
                continue
            
        
    def update_Z(self):
        '''
        update the value of Z, check line 6 of the ADMM algorithm for the math
        rho is lambda times rho2
        f is L_{rho}(W_t^{k+1}, b_t^{k+1}, g^{k+1}, (z_{ij}, z_{ji}, z_{(ij)^c}^k, u^k, h^k; t)
        see page 5 of https://arxiv.org/pdf/1703.07520.pdf Social discrete choice model
        '''
        for k in self.graph:
            for j in self.graph[k]:
                A = shared_array[j] + self.U[j][k]
                B = shared_array[k] + self.U[k][j]
                self.Z[k][j] = (2*self.Lambda*A + (2*self.Lambda+self.Rho)*B)/(self.Lambda*4+self.Rho)

    def update_U(self):
        '''
        update the value of U, check line 7 of the ADMM algorithm for the math
        '''
        for i in self.graph:
            for Id in self.graph[i]:
                self.U[i][Id] = self.U[i][Id] + shared_array[i] - self.Z[i][Id]

                
    '''
    using a simple gradient descent algorithm to update W
    https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
    learning rate is chosen using Backtracking linear search. see page 10 of the slides above
    '''
    def update_W(self, iteration):
        featureCnt = len(self.W)
        maxiter = 2
        oldloss = self.cal_LL()
        newloss = oldloss
        for k in range(maxiter):
            learningrate = 0.00001
            oldloss = newloss
            print('update W iteration {0}.{1}'.format(iteration, k))
            gradient = np.zeros(featureCnt)
            for i in self.graph:
                if (self.y[i]==0):
                    continue
                C1 = -1.0 * self.y[i]* (self.X[i].dot(self.W) + shared_array[i])
                eC1 = 0
                multiplier = 0
                try:
                    eC1 = math.exp(C1)
                except OverflowError:
                    eC1 = float('inf')
                if (eC1==float('inf')):
                    multiplier =  -1.0*self.y[i]
                else:
                    multiplier = (1 - 1.0/(1.0+eC1)) * (-1.0)*self.y[i]
                    
                gradient = np.add(gradient, multiplier * self.X[i])

            gradientNorm = np.linalg.norm(gradient)

            if (gradientNorm == float('inf')):
                raise ValueError('norm of gradient is infinity') #should never happen

            gradientNorm2 = gradientNorm * gradientNorm
            
            oldW = np.copy(self.W)
            kk=0
            newloss = 0
            tol = 1e-5
            while (True):
                np.copyto(self.W, oldW)
                self.W -= learningrate * gradient
                anticipateddecrease = learningrate * gradientNorm2 /2.0
                print('anticipate the loss to decrease from {0} by {1}'.format(oldloss, anticipateddecrease))
            
                targetloss = oldloss - anticipateddecrease
                
                try:
                    newloss = self.cal_LL()
                except OverflowError:
                    learningrate = learningrate / 2
                    kk+=1
                    print('get infinite loss, reduce learning rate to {0}, kk={1}'.format(learningrate, kk))
                    continue
                
                if (newloss <= targetloss + tol):
                    break;
                
                learningrate  = learningrate / 2
                
                kk+=1
                print('loss is not decreasing below anticipated value, reduce learning rate to {0}, kk={1}'.format(learningrate, kk))
                if(kk>1000):
                    raise ValueError('cannot find a good learning rate to get finite loss.learningrate={0}'.format(learningrate))
                
            print('learning rate: {0}'.format(learningrate))
            print('max in gradient is :{0}'.format(np.max(np.abs(gradient))))
            print('oldloss:' + str(oldloss) + ',newloss:' + str(newloss))
            if(math.fabs(newloss-oldloss) < 0.00001 * oldloss):
                return newloss
        return newloss
            
    def optimize_b(self, iterations, old_loss, verbose=False):
        kk = 0
        maxiter = 5
        while (True):
            start2 = time.time()
            # parallel
            jobs = []
            p = multiprocessing.Process(target=self.worker, args=(0, int(shared_array.shape[0]/2)))
            jobs.append(p)
            p.start()
            p = multiprocessing.Process(target=self.worker, args=(int(shared_array.shape[0]/2),int(shared_array.shape[0])))
            jobs.append(p)
            p.start()
            for proc in jobs:
                proc.join()
            # self.count_b = 0
            # start2 = time.time()
            # # parallel
            # jobs = []
            # pipe_list = []
            # recv_end, send_end = multiprocessing.Pipe(False)
            # print('start first half')
            # p = multiprocessing.Process(target=self.worker, args=(shared_array[0:int(shared_array.shape[0]/2)],0, send_end))
            # jobs.append(p)
            # pipe_list.append(recv_end)
            # p.start()
            # # 2
            # recv_end, send_end = multiprocessing.Pipe(False)
            # print('start second half')
            # p = multiprocessing.Process(target=self.worker, args=(shared_array[int(shared_array.shape[0]/2):int(shared_array.shape[0])],int(shared_array.shape[0]/2) ,send_end))
            # jobs.append(p)
            # pipe_list.append(recv_end)
            # p.start()
            # # 3



            # result_list = [x.recv() for x in pipe_list]
            # shared_array[0:int(shared_array.shape[0]/2)] = result_list[0]
            # shared_array[int(shared_array.shape[0]/2):int(shared_array.shape[0])] = result_list[1]
        
            

            end2 = time.time()
            if(verbose):
                print('finished b {0} seconds at iteration {1}'.format(end2-start2, iterations))
            start2 = time.time()
            self.update_Z()
            end2 = time.time()
            if(verbose):
                print('finished Z {0} seconds at iteration {1}'.format(end2-start2, iterations))
            start2 = time.time()
            self.update_U()
            end2 = time.time()
            if(verbose):
                print('finished U {0} seconds at iteration {1}'.format(end2-start2, iterations))
            loss = self.cal_LL()
            print('loss is {0}, old loss is {1} at iteration {2}.{3}'.format(loss, old_loss, iterations, kk))
            kk+=1
            if(np.absolute(old_loss-loss)<=self.Threshold):
                return loss
            if (kk > maxiter):
                return loss
                        
    def runADMM_Grid(self):
        '''
        runADMM Grid iterations
        The stopping criteria is when the difference of the value of the objective function in current iteration
        and the value of the objective function in the previous iteration is smaller than the Threshold
        '''
        resultdump = 'result.dump'
#         self.dumpWb(resultdump + ".initial")
        self.losses = []
        self.times = []
        loss = self.cal_LL()
        self.losses.append(loss)
        print('iteration = 0')
        print('objective = {0}'.format(loss))
        old_loss = loss
        loss = float('inf')
        iterations = 0
        import time
        start = time.time()
        self.times.append(0)
        while(True):
            # self.update_b()
            loss = self.optimize_b(iterations, old_loss)
            
            start2 = time.time()
            loss = self.update_W(iterations)
            end2 = time.time()
            print('finished w {0} seconds at iteration {1}'.format(end2-start2, iterations))
            print('loss is {0}, old loss is {1} at iteration {2}'.format(loss, old_loss, iterations))
            loss = self.cal_LL()
            self.losses.append(loss)
            end = time.time()
            duration = end - start
            print('Timing: ',duration)
            self.times.append(duration)
            # if(np.absolute(old_loss- loss) <=  self.Threshold):
            #     break
            # old_loss = loss
            # iterations += 1
            if(np.absolute(old_loss- loss) <=  self.Threshold and iterations>=3):
                break
            if(iterations>3):
                break
            # if(loss<69314.2467411):
            #     break
            old_loss = loss
            iterations += 1
#             if (iterations % 2 == 0):
#                 self.dumpWb(resultdump + "." + str(iterations))
            

        print('total iterations = ' + str(iterations))
        end = time.time()
        print('total time = {0}'.format(end-start))
#         self.dumpWb(resultdump + ".final" )

    def cal_LL(self):
        '''
        function to calculate the value of loss function
        '''
        W = np.array(self.W).flatten()
        b = np.array(shared_array).flatten()
        loss = 0
        for i in self.nodes:
            r = np.log(1 + np.exp(-self.y[i]*(np.dot(self.X[i], W) + b[i])))
            if(r == float('inf')):
                raise OverflowError('loss is infinity')
            loss += r
        
        for i, js in self.graph.items():
            for j in js:
                loss +=  self.Lambda*(shared_array[i]-shared_array[j])**2
        return loss