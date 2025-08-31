import numpy as np
# from .Quadratic import f
from .LeastSquares import f

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

def GradientDescent(A, Q, b, alpha=0.001, n=1000, tol=1e-5):
    x = np.zeros((A.shape[1], 1))
    costs = []
    x_list = [x.copy()]
    
    for i in range(n):
        grad = 2 * A.T @ Q @ (A @ x - b)  
        if np.linalg.norm(grad) < tol:
            break
        x = x - alpha * grad

        cost = f(A, Q, b, x)  # Cost 계산
        x_list.append(x.copy())
        costs.append(cost)

    return x_list, f(A, Q, b, x), costs

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

def SteepestDescent(A, Q, b, n=1000, tol=1e-5):
    x = np.zeros((A.shape[1], 1))
    costs = []
    x_list = [x.copy()]

    H = 2 * A.T @ Q @ A
    
    for i in range(n):
        grad = 2 * A.T @ Q @ (A @ x - b)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        alpha = (grad_norm**2) / (grad.T @ H @ grad) 
        x = x - alpha * grad

        cost = f(A, Q, b, x)
        x_list.append(x.copy())
        costs.append(cost.item())

    return x_list, f(A, Q, b, x), costs