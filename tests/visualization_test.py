import pytest
from piglegsurgeryweb.uploader import visualization_tools
import skimage.data
from matplotlib import pyplot as plt


def test_crop_square():
    astro = skimage.data.astronaut()
    astro = astro[:300, :]
    # plt.imshow(astro)
    # plt.show()

    astro = visualization_tools.crop_square(astro)
    # plt.imshow(astro)
    # plt.show()
    assert astro.shape[0] == astro.shape[1]
