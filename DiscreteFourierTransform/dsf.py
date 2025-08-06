import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# 예제 이미지 로드
image = misc.face(gray=True)

# 2D 푸리에 변환 수행
fft_image = np.fft.fft2(image)

# 주파수 성분을 중앙에 맞추기 위해 fftshift 적용
shifted_fft_image = np.fft.fftshift(fft_image)

# 원본 이미지와 푸리에 변환된 이미지 시각화
plt.subplot(2, 2, 1)
plt.imshow(np.log(np.abs(fft_image)), cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(shifted_fft_image)), cmap='gray')
plt.title('fftshift 후 이미지')

plt.show()