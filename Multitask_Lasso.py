def multitask_lasso(X, y, lr = 0.1, l1_ratio = 0.01, normalize = False, tempotal_smooth = True):
    n_samples, n_features = X.shape
    _, n_tasks = y.shape
    coef = np.zeros((n_tasks, n_features))
    if n_samples != y.shape[0]:
        print('X and y have inconsistent dimensions')
    