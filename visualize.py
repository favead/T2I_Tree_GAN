import matplotlib.pyplot as plt
import numpy as np


def show_image(img: np.ndarray) -> None:
    plt.imshow(img)
    plt.show()
