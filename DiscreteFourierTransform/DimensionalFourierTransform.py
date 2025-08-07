import numpy as np

def DFT(x):
    N = x.shape[0]
    W_N_kn = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            W_N_kn[k][n] = np.exp(-1j * (2 * np.pi * k / N) * n)

    X_N = W_N_kn @ x
    return X_N

def DFT_2d(x):
    M, N = x.shape
    x_ = np.zeros((M, N), dtype=complex)

    for i in range(M):
        x_[i, :] = DFT(x[i, :]).flatten()

    result = np.zeros((M, N), dtype=complex)
    for j in range(N):
        result[:, j] = DFT(x_[:, j]).flatten()

    return result

def InverseDFT(X):
    N = X.shape[0]
    W_N_nk = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            W_N_nk[n][k] = np.exp(1j * (2 * np.pi * k / N) * n)

    x_N = W_N_nk @ X
    return x_N / N

def InverseDFT_2d(x):
    M, N = x.shape
    x_ = np.zeros((M, N), dtype=complex)

    for i in range(M):
        x_[i, :] = InverseDFT(x[i, :]).flatten()

    result = np.zeros((M, N), dtype=complex)
    for j in range(N):
        result[:, j] = InverseDFT(x_[:, j]).flatten()

    return result

def LTI_Filter(x, coef = 1):
    return x * coef