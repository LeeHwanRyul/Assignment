import numpy as np
from .Quadratic import f

def GradientDescent(Q, b, alpha=0.001, n=1000, tol=1e-5):
    x = np.zeros((Q.shape[0], 1))
    costs = []
    x_list = [x.copy()]
    
    for i in range(n):
        grad = Q @ x - b
        if np.linalg.norm(grad) < tol:
            break
        x = x - alpha * grad

        cost = f(Q, b, x)
        x_list.append(x.copy())
        costs.append(cost.item())

    return x_list, f(Q, b, x), costs

def SteepestDescent(Q, b, n=1000, tol=1e-5):
    x = np.zeros((Q.shape[0], 1))
    costs = []
    x_list = [x.copy()]
    
    for i in range(n):
        grad = Q @ x - b
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        alpha = (grad_norm**2) / (grad.T @ Q @ grad) 
        x = x - alpha * grad

        cost = f(Q, b, x)
        x_list.append(x.copy())
        costs.append(cost.item())

    return x_list, f(Q, b, x), costs