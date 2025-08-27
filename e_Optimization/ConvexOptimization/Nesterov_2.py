import numpy as np
from .Quadratic import f

def Nesterov(Q, b, gamma=0.9, n=1000, tol=1e-5):
    x = np.zeros((Q.shape[0], 1))
    v = np.zeros((Q.shape[0], 1))
    costs = []
    
    for i in range(n):
        grad = Q @ x - b
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        alpha = (grad_norm**2) / (grad.T @ Q @ grad) 

        grad = Q @ (x - gamma * v) - b
        v = gamma * v + alpha * grad
        x = x - v

        cost = f(Q, b, x)
        costs.append(cost.item())

    return x, f(Q, b, x), costs