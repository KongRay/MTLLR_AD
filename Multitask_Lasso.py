import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
X = pd.read_excel()
y = pd.read_excel()
def multitask_lasso(X, y, lr = 0.1, alpha = 0.01, max_iter = 100, min_gap = 0.001, normalize = False, tempotal_smooth = True):
    loss_train_record = []
    loss_dev_record = []
    weights = np.zeros((y.shape[1], X.shape[1]))
    #Set necessary factors

    if normalize == True:
        X = (X - X.mean())/X.std()
    X = np.array(X)
    y = np.array(y)
    #Data pre-processing

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.3)
    # Randomly split the dataset

    for iter in range(max_iter):
        if tempotal_smooth == True:
            weights = weights - lr * np.dot(np.dot(X_train, weights.T) - y_train, X_train) - 2 * lr * alpha * weights
            loss_train = ((y_train - np.dot(X_train, weights.T)) ** 2).sum(axis = 0) + alpha * (weights ** 2).sum()
            loss_dev = ((y_dev - np.dot(X_dev, weights.T)) ** 2).sum(axis = 0) + alpha * (weights ** 2).sum()
            #Need to add Temporal Smoothness Prior
        else:
            weights = weights - lr * np.dot(np.dot(X_train, weights.T) - y_train, X_train) - 2 * lr * alpha * weights
            loss_train = ((y_train - np.dot(X_train, weights.T)) ** 2).sum(axis = 0) + alpha * (weights ** 2).sum()
            loss_dev = ((y_dev - np.dot(X_dev, weights.T)) ** 2).sum(axis = 0) + alpha * (weights ** 2).sum()
            #Realize min ||XW−Y||^2 + θ1||W||^2
        loss_train_record.append(loss_train)
        loss_dev_record.append(loss_dev)
        if iter != 0:
            if (loss_dev_record[iter] - loss_dev_record[iter - 1]) < min_gap:
        print(iter, loss_train, loss_dev)
    #Train the Lasso Regression
    return weights

prediction = np.dot(X_test, weights.T)
#Need to add evaluation methods