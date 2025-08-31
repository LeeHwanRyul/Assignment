import numpy as np
from .Quadratic import f
from .LeastSquares import f

def Nesterov(Q, b, gamma=0.9, alpha=0.0001, n=1000, tol=1e-5):
    x = np.zeros((Q.shape[0], 1))
    v = np.zeros((Q.shape[0], 1))
    costs = []
    x_list = [x.copy()]
    
    for i in range(n):
        grad = Q @ x - b
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        grad = Q @ (x - gamma * v) - b
        v = gamma * v + alpha * grad
        x = x - v

        cost = f(Q, b, x)
        x_list.append(x.copy())
        costs.append(cost.item())

    return x_list, f(Q, b, x), costs

def Nesterov(A, Q, b, gamma=0.9, alpha=0.0001, n=1000, tol=1e-5):
    x = np.zeros((A.shape[1], 1))
    v = np.zeros((A.shape[1], 1))
    costs = []
    x_list = [x.copy()]
    
    for i in range(n):
        x_lookahead = x - gamma * v

        # Gradient at look-ahead point
        grad = 2 * A.T @ Q @ (A @ x_lookahead - b)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        v = gamma * v + alpha * grad
        x = x - v

        cost = f(A, Q, b, x)
        x_list.append(x.copy())
        costs.append(cost.item())

    return x_list, f(A, Q, b, x), costs