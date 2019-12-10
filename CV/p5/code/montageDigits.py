import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage2d

def montageDigits(x):
    num_images = x.shape[2]
    m = montage2d(x.transpose(2, 0, 1))
    plt.imshow(m, cmap='gray')
    plt.show()

    return np.mean(x, axis=2)
