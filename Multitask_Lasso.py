import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
X = pd.read_csv()
y = pd.read_csv()
X = np.array(X)
y = np.array(y)
def multitask_lasso(X, y, lr = 0.1, alpha = 0.01, alpha_t = 0.01, max_iter = 100, min_gap = 0.001, tempotal_smooth = False, incomplete_data = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.3)
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
    S = np.zeros((y_train.shape[0], y_train.shape[1]))
    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            if y[i,j] != 0:#if the data exists
                S[i,j] = 1
    #Set S for dealing with incomplete data

    for iter in range(max_iter):
        if tempotal_smooth == True & incomplete_data == False:
            weights = weights - lr * (np.dot(np.dot(X_train.T, X_train), weights)-np.dot(X_train.T, y_train) + alpha * weights + alpha_t * np.dot(np.dot(weights,H),H.T))
            loss_train = np.linalg.norm(np.dot(X_train, weights)-y_train)**2 + alpha * np.linalg.norm(weights)**2 + alpha_t * np.linalg.norm(np.dot(weights,H))**2
            loss_dev = np.linalg.norm(np.dot(X_dev, weights)-y_dev)**2 + alpha * np.linalg.norm(weights)**2
            #Only add Temporal Smoothness Prior
        elif tempotal_smooth == True & incomplete_data == True:
            weights = weights - lr * (S*(np.dot(np.dot(X_train.T, X_train), weights)-np.dot(X_train.T, y_train)) + alpha * weights + alpha_t * np.dot(np.dot(weights,H),H.T))
            loss_train = np.linalg.norm(S*(np.dot(X_train, weights)-y_train))**2 + alpha * np.linalg.norm(weights)**2 + alpha_t * np.linalg.norm(np.dot(weights,H))**2
            loss_dev = np.linalg.norm(S*(np.dot(X_dev, weights)-y_dev))**2 + alpha * np.linalg.norm(weights)**2
            #Add both TSP and Incomplete Data
        else:
            weights = weights - lr * (np.dot(np.dot(X_train.T, X_train),weights)-np.dot(X_train.T, y_train) + alpha * weights)
            loss_train = np.linalg.norm(np.dot(X_train, weights)-y_train)**2 + alpha * np.linalg.norm(weights)**2
            loss_dev = np.linalg.norm(np.dot(X_dev, weights)-y_dev)**2 + alpha * np.linalg.norm(weights)**2
            #Realize min ||XW−Y||F^2 + θ1||W||F^2
        loss_train_record.append(loss_train)
        loss_dev_record.append(loss_dev)
        #Collect loss for each iteration
        if iter != 0:
            if (loss_dev_record[iter] - loss_dev_record[iter - 1]) < min_gap:
                break
    #Train the Multitask Lasso Regression by gradient descent
    return weights, X_test, y_test

weights, X_test, y_test = multitask_lasso(X, y, lr = 0.1, alpha = 0.01, alpha_t = 0.01, max_iter = 100, min_gap = 0.001, tempotal_smooth = True)
prediction = np.dot(X_test, weights.T)
#Need to add evaluation methods