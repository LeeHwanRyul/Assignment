import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from DimensionalFourierTransform import DFT, InverseDFT, DFT_2d, InverseDFT_2d

if __name__ == "__main__":
    x = Image.open('livingroom.tif').convert('L')
    x = np.array(x, dtype=np.float64)

    x_dft2d = DFT_2d(x)
    x_fft2 = np.fft.fft2(x)

    # 4.
    E = x_dft2d - x_fft2
    E_hermitian = E.conj().T @ E
    E_det = np.linalg.det(E_hermitian)

    print("E_det: {}".format(E_det))

    # 5.
    data = [['DFT 2d', np.log1p(np.abs(x_dft2d))],
            ['fft2', np.log1p(np.abs(x_dft2d))],
            ['shift DFT 2d', np.log1p(np.abs(np.fft.fftshift(x_dft2d)))],
            ['shift fft2', np.log1p(np.abs(np.fft.fftshift(x_fft2)))]]

    for i in range(len(data)):
        plt.subplot(3, 2, i + 3)
        plt.imshow(data[i][1], cmap='gray')
        plt.title(data[i][0])
        plt.colorbar()

    # 6.
    x_idft = InverseDFT_2d(x_dft2d)
    F = x_idft - x
    F_hermitian = F.conj().T @ F
    F_det = np.linalg.det(F_hermitian)

    print("F_det: {}".format(F_det))

    plt.subplot(3, 2, 1)
    plt.imshow(x, cmap='gray')
    plt.title('original image')
    plt.colorbar()

    plt.subplot(3, 2, 2)
    plt.imshow(x_idft, cmap='gray')
    plt.title('inverse DFT image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()