import numpy as np


def psnr(x, y, maximum=1.0):
    return 20 * np.log10(maximum / np.sqrt(np.mean((x - y) ** 2)))
