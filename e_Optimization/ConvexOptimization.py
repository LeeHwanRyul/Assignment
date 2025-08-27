import numpy as np
from ConvexOptimization import Convexity, GenerateMatrix

if __name__ == "__main__":
    n = 1000
    rho = 0.001

    Q = GenerateMatrix(n, rho)
    b = np.random.randn(n, 1)
    t = 1000
    
    # 1. Convexity of function 
    # An Introduction to Optimization 
    # af(u)+(1-a)f(u) >= f(au+(1-a)v)를 만족하는 f
    if Convexity(Q, b, t):
        print("1. function is convexity")
    else:
        print("Error: function isn't convexity")
        exit()

    