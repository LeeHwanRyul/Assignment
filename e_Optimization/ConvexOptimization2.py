import numpy as np
import cvxpy as cp
from ConvexOptimization import Convexity, GenerateMatrix
from ConvexOptimization import OptimalSolution
from ConvexOptimization import GradientDescent, SteepestDescent
from ConvexOptimization import Nesterov
import matplotlib.pyplot as plt

if __name__ == "__main__":
    listGD, listSD, listNAG = [], [], []
    l = 1000
    rho = 5

    K = [[50, 100], [100, 50]]

    for m, n in K:
        A = np.random.randn(m, n)
        Q = GenerateMatrix(m, rho)
        b = np.random.randn(m, 1)
        t = 1000

        print("m =",m)
        print("n =",n)

        # 1. Gradient Solve
        # f(x) = (Ax-b)'Q(Ax-b)
        # nabla f(x) = pf/px = 2A'Q(Ax-b)
        # so 2A'Q(Ax^*-b)=0 -> A'QAx^* = A'Qb
        # x^* = (A'QA)^(-1)A'Qb
        x_star = np.linalg.pinv(A.T @ Q @ A) @ (A.T @ Q @ b)
        r = A @ x_star - b
        f_star = float(r.T @ Q @ r)

        print("1. Obtimal Solution by Hand:", f_star)

        # 2. Gradient Descent
        alpha = 0.00001
        n = 10000
        optSol_GD, optCost_GD, GD_plot = GradientDescent(A, Q, b, alpha, n)
        print("2. Obtimal Solution of GradientDescent:", optCost_GD)
        
        n = 10000
        optSol_SD, optCost_SD, SD_plot = SteepestDescent(A, Q, b, n)
        print("3. Obtimal Solution of SteepestDescent:", optCost_GD)

        gamma = 0.9
        optSol_NAG, optCost_NAG, NAG_plot = Nesterov(A, Q, b, gamma, alpha, n)
        print("4. Obtimal Solution of Nesterov-2:", optCost_NAG)