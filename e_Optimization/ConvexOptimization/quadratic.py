import numpy as np

def GenerateMatrix(n, rho):
    A = np.random.randn(n, n)
    return A @ A.T + rho * np.eye(n)

def f(Q, b, x):
    return 0.5 * x.T @ Q @ x - b.T @ x

def Convexity(Q, b, trials=5):
    n = Q.shape[0]
    for _ in range(trials):
        u = np.random.randn(n, 1)
        v = np.random.randn(n, 1)
        a = np.random.rand()


        if a*f(Q,b,u)+(1-a)*f(Q,b,v) >= f(Q,b,a*u+(1-a)*v):
            return True
    return False