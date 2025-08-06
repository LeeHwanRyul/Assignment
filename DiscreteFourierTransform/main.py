from DimensionalFourierTransform import DFT, InverseDFT
import numpy as np

if __name__ == "__main__":
    x = np.array([100, 0, 100, 0, 50]).reshape(5, 1)
    print(InverseDFT(DFT(x)))