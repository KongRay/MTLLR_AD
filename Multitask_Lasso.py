import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
X = pd.read_csv('MRI-preprocess.csv')
y = pd.read_csv('CSF_task_mode.csv')
X = np.array(X)#(n_samples, n_features)
y = np.array(y)#(n_samples, n_tasks)

def multitask_lasso(X, y, lr = 0.1, alpha = 0.01, alpha_t = 0.01, max_iter = 100, min_gap = 0.001, tempotal_smooth = False, incomplete_data = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.3)
    print('finish split')
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
    print('finish H')
    #Set H matrix for TSP
    S = np.zeros((y_train.shape[0], y_train.shape[1]))
    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            if math.isnan(y[i,j]) == False:#if the data exists
                S[i,j] = 1
    print('finish S')
    #Set S for dealing with incomplete data

    for iter in range(max_iter):
        print(iter)
        if tempotal_smooth == True & incomplete_data == False:
            weights = weights - lr * (np.dot(np.dot(X_train.T, X_train), weights)-np.dot(X_train.T, y_train) + alpha * weights + alpha_t * np.dot(np.dot(weights,H),H.T))
            loss_train = np.linalg.norm(np.dot(X_train, weights)-y_train)**2 + alpha * np.linalg.norm(weights)**2 + alpha_t * np.linalg.norm(np.dot(weights,H))**2
            loss_dev = np.linalg.norm(np.dot(X_dev, weights)-y_dev)**2 + alpha * np.linalg.norm(weights)**2
            print(loss_train)
            print(loss_dev)
            #Only add Temporal Smoothness Prior
            #minW ||XW−Y||F^2 + θ1||W||F^2 + θ2||WH||F^2
            #∂J(W)/∂W = X.T * X * W -X.T * y + θ1 * W + θ2 * W * H * H.T
        elif tempotal_smooth == True & incomplete_data == True:
            weights = weights - lr * (S*(np.dot(np.dot(X_train.T, X_train), weights)-np.dot(X_train.T, y_train)) + alpha * weights + alpha_t * np.dot(np.dot(weights,H),H.T))
            loss_train = np.linalg.norm(S*(np.dot(X_train, weights)-y_train))**2 + alpha * np.linalg.norm(weights)**2 + alpha_t * np.linalg.norm(np.dot(weights,H))**2
            loss_dev = np.linalg.norm(S*(np.dot(X_dev, weights)-y_dev))**2 + alpha * np.linalg.norm(weights)**2
            print(loss_train)
            print(loss_dev)
            #Add both TSP and Incomplete Data
            #minW ||S⊙(XW−Y)||F^2 + θ1||W||F^2 + θ2||WH||F^2
            #∂J(W)/∂W = S * (X.T * X * W -X.T * y) + θ1 * W + θ2 * W * H * H.T
        else:
            weights = weights - lr * (np.dot(np.dot(X_train.T, X_train),weights)-np.dot(X_train.T, y_train) + alpha * weights)
            loss_train = np.linalg.norm(np.dot(X_train, weights)-y_train)**2 + alpha * np.linalg.norm(weights)**2
            loss_dev = np.linalg.norm(np.dot(X_dev, weights)-y_dev)**2 + alpha * np.linalg.norm(weights)**2
            print(loss_train)
            print(loss_dev)
            #Realize minW ||XW−Y||F^2 + θ1||W||F^2
            #∂J(W)/∂W = X.T * X * W -X.T * y + θ1 * W
        loss_train_record.append(loss_train)
        loss_dev_record.append(loss_dev)
        #Collect loss for each iteration
        if iter != 0:
            if (loss_dev_record[iter] - loss_dev_record[iter - 1]) < min_gap:
                break
    #Train the Multitask Lasso Regression by gradient descent
    return weights, X_test, y_test, loss_train_record, loss_dev_record

weights, X_test, y_test, trl, devl = multitask_lasso(X, y, lr = 0.1, alpha = 0.01, alpha_t = 0.01, max_iter = 100, min_gap = 0.001, tempotal_smooth = False, incomplete_data = False)
#prediction = np.dot(X_test, weights)
#Need to add evaluation methods