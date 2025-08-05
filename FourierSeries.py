import numpy as np

def FourierSeries(t, N, func, amp, T, eulerCoef = 500):
    # 기본 위상각
    omega_o = 2 * np.pi / T

    # 오일러 적분
    def EulerIntegral(a, b, k):
        sum = 0
        deltaT = ((b - a) / eulerCoef)

        for i in range(0, eulerCoef):
            t_k = a + i * deltaT
            sum += func(t_k, amp, T) * np.exp(-1j * k * omega_o * t_k) * deltaT
        return sum

    sum = np.zeros_like(t, dtype=complex)

    # C_k의 진폭과 위상
    CList = []
    for k in range(-N, N+1):
        C_k = EulerIntegral(0, T, k) / T
        sum += C_k * np.exp(1j * k * omega_o * t)
        CList.append(C_k)

    return np.real(sum), CList