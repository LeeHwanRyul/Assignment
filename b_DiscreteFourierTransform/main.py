import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from DimensionalFourierTransform import DFT, InverseDFT, DFT_2d, InverseDFT_2d, LTI_Filter

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

    # 6.
    x_idft = InverseDFT_2d(x_dft2d)
    F = x_idft - x
    F_hermitian = F.conj().T @ F
    F_det = np.linalg.det(F_hermitian)

    print("F_det: {}".format(F_det))

    x_lti_3 = LTI_Filter(x_dft2d, 2.5)
    x_lti_01 = LTI_Filter(x_dft2d, 0.3)

    x_idft_3 = InverseDFT_2d(x_lti_3)
    x_idft_01 = InverseDFT_2d(x_lti_01)

    data = [['original image', x],
            ['inverse DFT image', np.real(x_idft)],
            ['shift DFT 2d', np.log1p(np.abs(np.fft.fftshift(x_dft2d)))],
            ['shift fft2', np.log1p(np.abs(np.fft.fftshift(x_fft2)))]]

    for i in range(len(data)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(data[i][1], cmap='gray')
        plt.title(data[i][0])
        plt.colorbar()

    plt.tight_layout()
    plt.show()

    # 실수부만 취하고, 반올림 후 uint8로 변환
    x_idft_uint8 = np.round(np.real(x_idft_3)).astype(np.uint8)

    # PIL 이미지 객체로 변환 후 저장
    image = Image.fromarray(x_idft_uint8)
    image.save("livingroom_idft_3_reconstructed.tiff")

    # 실수부만 취하고, 반올림 후 uint8로 변환
    x_idft_uint8 = np.round(np.real(x_idft_01)).astype(np.uint8)

    # PIL 이미지 객체로 변환 후 저장
    image = Image.fromarray(x_idft_uint8)
    image.save("livingroom_idft_01_reconstructed.tiff")