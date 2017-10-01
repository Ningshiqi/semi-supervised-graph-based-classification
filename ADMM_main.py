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
# all the class objects are in the Model.py file
from ADMM import *
from ADMM_model_9 import *
import OutputHelper as sp



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Lambda', type=float, default=10,
        help='hyperparameter to control graph regularization')
    parser.add_argument('--Rho', type=float, default=1.0,
        help='hyperparameter to control ADMM stepsize')
    parser.add_argument('--Threshold', type=float, default=0.1,
        help='hyperparameter as the stopping criteria of ADMM algorithm')
    parser.add_argument('--ADMM_output_file', type=str, default="C9.p",
        help='Name of output file of ADMM class')
    parser.add_argument('--plot_file', type=str, default="run1.png",
        help='Name of losses versus iteration plot')
    parser.add_argument('--data_file', type=str, default="run1.p",
        help='Name of saved values')  
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--train_nodes", type=int, default=200)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument('--prob', type=float, default=0.1)
    args = parser.parse_args()
    # X, temp_new, Y_train, expamountDict, nodes, edgecnt,Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U = generate_data(args.nodes, args.train_nodes, args.prob) 
    # list_data = [X, temp_new, Y_train, expamountDict, nodes, edgecnt,Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U]
    # pickle.dump( list_data, open( "100000.p", "wb" ) ) 
    file = open("data/10000_1.p", 'rb')
    list_data = pickle.load(file)
    file.close()
    [X, temp_new, Y_train, expamountDict, nodes, edgecnt,Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U] = list_data
    # print(type(temp_new))
    print('import data done')
    print('number of nodes: ',len(nodes))
    print('number of edges: ', edgecnt)
    print('X shape', X.shape)
    # print(nx.number_of_nodes(temp_new))
    # print(temp_new.number_of_nodes())
    # print(temp_new.number_of_edges())

    processes = args.processes
    Lambda = args.Lambda
    Rho = args.Rho
    Threshold = args.Threshold


    # print('..............')
    # print('run centrailized version')
    # A = ADMM(X, Y_train, expamountDict, nodes, edgecnt, Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U)
    # start = time.time()
    # A.runADMM_Grid()
    # end = time.time()
    # print(end- start)
    # print(A.times)
    # print(A.losses)

    print('.................')
    print('two processors')
    E = ADMM_par(X, Y_train, expamountDict, nodes, edgecnt, Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U)
    start = time.time()
    E.runADMM_Grid()
    end = time.time()
    print(end- start)
    print(E.times)
    print(E.losses)

    
    # print('.................')
    # print('two processors')
    # D = ADMM_par(X, Y_train, expamountDict, nodes, edgecnt, Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U)
    # start = time.time()
    # D.runADMM_Grid()
    # end = time.time()
    # print(end- start)
    # print(D.times)
    # print(D.losses)
    
    # # F = ADMM_par(X, Y_train, expamountDict, nodes, edgecnt, Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U, 5)
    # # start = time.time()
    # # F.runADMM_Grid()
    # # end = time.time()
    # # print(end- start)

    # # C = ADMM_par(X, Y_train, expamountDict, nodes, edgecnt, Lambda, Rho, train_mask,test_mask,Y_true, Threshold, coef, intercept, Z, U, 8)
    # # start = time.time()
    # # C.runADMM_Grid()
    # # end = time.time()
    # # print(end- start)



    # # export figure
    # fig = plt.figure(3,figsize=(10,6))
    # ax = plt.subplot(111)
    # # ax.plot(C.times[1:],C.losses[1:],label='8 processors')
    # # ax.scatter(C.times[1:], C.losses[1:], c='r')
    # ax.plot(D.times[1:],D.losses[1:],label='1 processors')
    # ax.scatter(D.times[1:], D.losses[1:], c='r')
    # ax.plot(E.times[1:],E.losses[1:],label='3 processors')
    # ax.scatter(E.times[1:], E.losses[1:], c='r')
    # # ax.plot(F.times[1:],F.losses[1:],label='5 processors')
    # # ax.scatter(F.times[1:], F.losses[1:], c='r')
    # plt.title('Training losses',fontsize=25)
    # plt.legend(fontsize =15)
    # fig.savefig(args.plot_file)
    # plt.close()
    # pickle.dump( [D.times, D.losses, E.times, E.losses], open( args.data_file, "wb" ) )
    #pickle.dump( [C.times, C.losses, D.times, D.losses, E.times, E.losses, F.times, F.losses], open( args.data_file, "wb" ) )


