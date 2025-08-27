import numpy as np
from ConvexOptimization import Convexity, GenerateMatrix
from ConvexOptimization import OptimalSolution
from ConvexOptimization import GradientDescent, SteepestDescent
from ConvexOptimization import Nesterov
import matplotlib.pyplot as plt

if __name__ == "__main__":
    listGD, listSD, listNAG = [], [], []
    n = 1000
    rho = 5

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

    # 2. CVXPY package
    # Obtimal Solution 
    optSol_CVXPY, optCost_CVXPY = OptimalSolution(Q, b, n)
    print("2. Obtimal Solution of CVXPY:", optCost_CVXPY)

    # 3. Gradient Descent algorithm
    # x_k+1 = x_k - alpha * nabla f(x_k)
    alpha = 0.0001
    n = 10000
    optSol_GD, optCost_GD, GD_plot = GradientDescent(Q, b, alpha, n)
    print("3. Obtimal Solution of GradientDescent:", optCost_GD)

    # 4. Steepest Gradient Descent algorithm
    # x_k+1 = x_k - alpha * nabla f(x_k)
    # alpha = ||nabla f(x_k)||^2 / nabla f(x_k).T @ Q @ nabla f(x_k)
    n = 10000
    optSol_SD, optCost_SD, SD_plot = SteepestDescent(Q, b, n)
    print("4. Obtimal Solution of SteepestDescent:", optCost_GD)

    # 5. Nesterov-2 algorithm
    # v_t = gamma*v_{t-1}+alpha*nabla*J(theta-gamma*v_{t-1})
    # gamma: momentum 계수
    # alpha: learning rate
    gamma = 0.9
    alpha = 0.0001
    n = 10000
    optSol_NAG, optCost_NAG, NAG_plot = Nesterov(Q, b, gamma, alpha, n)
    print("3. Obtimal Solution of GradientDescent:", optCost_NAG)

    # 6. plot
    gd_norms = [optSol_CVXPY - x for x in optSol_GD]
    sd_norms = [optSol_CVXPY - x for x in optSol_SD]
    nag_norms = [optSol_CVXPY - x for x in optSol_NAG]

    print(gd_norms)

    plt.plot(GD_plot, label="Gradient Descent")
    plt.plot(SD_plot, label="Steepest Descent")
    plt.plot(NAG_plot, label="Nesterov-2 Descent")
    plt.legend()

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    axs[0].plot(gd_norms, label="Gradient Descent")
    axs[0].legend()

    axs[1].plot(sd_norms, label="Steepest Descent")
    axs[1].legend()

    axs[2].plot(nag_norms, label="Nesterov-2 Descent")
    axs[2].legend()

    plt.tight_layout()
    plt.show()