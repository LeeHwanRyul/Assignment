import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from NumericalAnalysis import EulerMethod
def RiccatiEquation(t, x):
    result = 2 * x - (x ** 2) / 2 + 1
    return result

if __name__ == "__main__":
    t = np.linspace(0, 1, 1000)

    x_0 = 1

    x_euler = EulerMethod(RiccatiEquation, t, x_0)

    sol = solve_ivp(RiccatiEquation, [t[0], t[-1]], [x_0], t_eval=t)
    x_ref = sol.y[0]

    # 오차 계산
    error = np.abs(x_ref - x_euler)

    # 시각화
    plt.plot(t, x_ref, label='solve_ivp (accurate)')
    plt.plot(t, x_euler, '--', label='Euler Method')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Max Error:", np.max(error))