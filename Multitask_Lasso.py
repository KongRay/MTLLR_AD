import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
X = pd.read_csv('MRI-preprocess.csv')
y = pd.read_csv('CSF_task_mode.csv')
X = (X - X.mean())/X.std()
y = (y - y.mean())/y.std()
X = np.array(X)#(n_samples, n_features)
y = np.array(y)#(n_samples, n_tasks)

def incomplete_data(y):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if math.isnan(y[i,j]) == True:#if the data doesn't exists
                y[i,j] = 0
    return y

def multitask_lasso(X, y, lr = 0.1, alpha = 0.01, alpha_t = 0.01, max_iter = 100,
                         min_gap = 0.001, temporal_smooth = False, analytic_expression = False):
    X_train_pre, X_test, y_train_pre, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_pre, y_train_pre, test_size = 0.2, random_state=42)
    # Randomly split the dataset

    loss_train_record = []
    loss_dev_record = []
    weights = np.zeros((X.shape[1], y.shape[1]))
    #Set necessary factors
    H = np.zeros((y.shape[1], y.shape[1]-1))
    for i in range(y.shape[1]):
        for j in range(y.shape[1]-1):
            if i == j:
                H[i,j] = 1
            elif i == j + 1:
                H[i,j] = -1
    #Set H matrix for TSP
    y_train_pre = incomplete_data(y_train_pre)
    y_train = incomplete_data(y_train)
    y_dev = incomplete_data(y_dev)
    #Dealing the Nan data of output
    

    for iter in range(max_iter):
        if (temporal_smooth == True and analytic_expression == False):
            weights = weights - lr * (np.dot(np.dot(X_train.T, X_train), weights)-np.dot(X_train.T, y_train) + 2 * alpha * weights + alpha_t * np.dot(np.dot(weights,H),H.T))
            loss_train = np.linalg.norm((np.dot(X_train, weights)-y_train)**2) + alpha * np.linalg.norm(weights **2) + alpha_t * np.linalg.norm(np.dot(weights,H)**2)
            loss_dev = np.linalg.norm((np.dot(X_dev, weights)-y_dev)**2) + alpha * np.linalg.norm(weights**2) + alpha_t * np.linalg.norm(np.dot(weights,H)**2)
            print(iter+1,loss_train, loss_dev)
            #Only add Temporal Smoothness Prior
            #minW ||XW−Y||F^2 + θ1||W||F^2 + θ2||WH||F^2
            #∂J(W)/∂W = X.T * X * W -X.T * y + θ1 * W + θ2 * W * H * H.T
        elif (temporal_smooth == True and analytic_expression == True):
            m1 = np.dot(X_train_pre.T, X_train_pre) + alpha * np.identity(X_train_pre.shape[1])
            m2 = alpha_t * np.dot(H,H.T)
            derta1, Q1 = np.linalg.eig(m1)
            derta2, Q2 = np.linalg.eig(m2)
            D = np.dot(np.dot(np.dot(Q1.T, X_train_pre.T), y_train_pre), Q2)
            W = np.zeros((derta1.shape[0], derta2.shape[0]))
            for i in range(derta1.shape[0]):
                for j in range(derta2.shape[0]):
                    W[i][j] = D[i][j]/(derta1[i]+derta2[j])
            weights = np.dot(np.dot(Q1, W), Q2.T)
            loss_train = np.linalg.norm((np.dot(X_train_pre, weights)-y_train_pre)**2) + alpha * np.linalg.norm(weights**2)+ alpha_t * np.linalg.norm(np.dot(weights,H)**2)
            print(loss_train)
            #realize Temporal Smoothness Prior with analytic expression
            #Q and derta are eigen vectors and eigen values matrix of (X.T * X + θ1 * I) and (θ2 * H * H.T)
            #D = Q1.T * X.T * Y * Q2
            #Wij = Dij / (derta1i + derta2j)
        elif (temporal_smooth == False and analytic_expression == False):
            weights = weights - lr * (np.dot(np.dot(X_train.T, X_train),weights)-np.dot(X_train.T, y_train) + 2 * alpha * weights)
            loss_train = np.linalg.norm((np.dot(X_train, weights)-y_train)**2) + alpha * np.abs(weights).sum()
            loss_dev = np.linalg.norm((np.dot(X_dev, weights)-y_dev)**2) + alpha * np.abs(weights).sum()
            print(iter+1,loss_train, loss_dev)
            #Realize minW ||XW−Y||F^2 + θ1||W||F^2
            #∂J(W)/∂W = X.T * X * W -X.T * y + 2 * θ1 * W
        else:
            ae1 = np.dot(X_train_pre.T, X_train_pre) + alpha * np.identity(X_train_pre.shape[1])
            ae2 = np.dot(X_train_pre.T, y_train_pre)
            weights = np.dot(np.linalg.inv(ae1), ae2)
            loss_train = np.linalg.norm((np.dot(X_train_pre, weights)-y_train_pre)**2) + alpha * np.abs(weights).sum()
            print(loss_train)
            #W = (X.T * X + θ1 * I)^(-1) * X.T * Y
        if analytic_expression == True:
            loss_train_record.append(loss_train)
            break
        else:
            loss_train_record.append(loss_train)
            loss_dev_record.append(loss_dev)
        #Collecting the loss history
        if iter != 0:
            if (loss_dev_record[iter-1] - loss_dev_record[iter]) < min_gap:
                break
    #Train the Multitask Lasso Regression by gradient descent
    return weights, X_test, y_test, loss_train_record, loss_dev_record

weights, X_test, y_test, trl, devl = multitask_lasso(X, y, lr = 0.001, alpha = 0.01, alpha_t = 0.01, max_iter = 100, min_gap = 0.001, temporal_smooth = False, analytic_expression = False)
#prediction = np.dot(X_test, weights)
#Need to add evaluation methods