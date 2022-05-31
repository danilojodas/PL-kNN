# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:33:52 2021

@author: DANILO
"""

import numpy as np
from pl_nn.pl_nn import PlNearestNeighbors
from m_knn.m_knn import MKNearestNeighbors

def calculate_accuracy(y_true,y_pred):
    return sum(y_pred == y_true) / len(y_true)

###############################################################################
# Loading the data
data = np.loadtxt('data/wine.txt',delimiter=',',dtype=float)
X = data[:,1:].astype(float)
y = data[:,0].astype(int)
###############################################################################


###############################################################################
# Split the data into train (80%) and test (20%)
random_list = list(range(X.shape[0]))
np.random.shuffle(random_list)

train_idx = random_list[0:round(len(random_list)*0.8)]
test_idx = random_list[0:round(len(random_list)*0.2)]

X_train,y_train = X[train_idx,:],y[train_idx]
X_test,y_test = X[test_idx,:],y[test_idx]
###############################################################################


###############################################################################
# Scaling the data according to a normal distribution
X_train_sc = (X_train - np.mean(X_train,axis=0)) / np.std(X_train,axis=0)
X_test_sc = (X_test - np.mean(X_test,axis=0)) / np.std(X_test,axis=0)
###############################################################################


###############################################################################
# Performing the training
plnn = PlNearestNeighbors()
smknn = MKNearestNeighbors(mode='smknn')
lmknn = MKNearestNeighbors(mode='lmknn')
plnn.fit(X_train_sc, y_train)
smknn.fit(X_train_sc, y_train)
lmknn.fit(X_train_sc, y_train)
###############################################################################


###############################################################################
# Performing predictions
y_pred_1 = plnn.predict(X_test_sc)
y_pred_2 = smknn.predict(X_test_sc)
y_pred_3 = lmknn.predict(X_test_sc)
###############################################################################

###############################################################################
# Computing accuracies
y_test=y_test.reshape(-1,1)
print('Accuracy PL-NN: ',calculate_accuracy(y_test, y_pred_1))
print('Accuracy SMKNN: ',calculate_accuracy(y_test, y_pred_2))
print('Accuracy LMKNN: ',calculate_accuracy(y_test, y_pred_3))
###############################################################################