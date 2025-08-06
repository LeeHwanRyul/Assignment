import numpy as np

def InverseDFT(X):
    N = X.shape[0]
    W_N_nk = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for n in range(N):
            W_N_nk[n][k] = np.exp(1j * (2 * np.pi * k / N) * n)

    x_N = W_N_nk @ X
    return x_N / N

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

    dft1 = np.zeros_like(x, )