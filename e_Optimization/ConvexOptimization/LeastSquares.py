import numpy as np

def GenerateMatrix(n, rho):
    A = np.random.randn(n, n)
    return A @ A.T + rho * np.eye(n)

def f(A, Q, b, x):
    return (A @ x - b).T @ Q @ (A @ x - b)

def Convexity(A, Q, b, trials=5):
    n = A.shape[1]
    for _ in range(trials):
        u = np.random.randn(n, 1)
        v = np.random.randn(n, 1)
        a = np.random.rand()

        if a*f(A,Q,b,u)+(1-a)*f(A, Q,b,v) >= f(A, Q,b,a*u+(1-a)*v):
            return True
    return False