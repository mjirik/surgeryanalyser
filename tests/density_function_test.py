import skimage.io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from piglegcv import run_report
import skimage.data
from pathlib import Path


def test_density():
    nsamples = 100
    data1 = np.random.normal([10, 6], [5, 4], size=[nsamples, 2])
    data2 = np.random.normal([60, 20], [10, 7], size=[nsamples, 2])
    data = np.concatenate([data1, data2], axis=0)
    print(data.shape)

    plt.plot(data[:, 0], data[:, 1], "*")
    df = pd.DataFrame(data, columns=["x", "y"])

    sns.kdeplot(data=df, x="x", y="y", fill=True)

    # plt.show()


def test_heatmap():
    astro = skimage.data.astronaut()
    nsamples = 100
    data1 = np.random.normal([100, 160], [50, 40], size=[nsamples, 2])
    data2 = np.random.normal([160, 200], [10, 70], size=[nsamples, 2])
    data = np.concatenate([data1, data2], axis=0)

    fn = Path("heatmap.png")
    if fn.exists():
        fn.unlink()

    fig = run_report.create_heatmap_report(data, image=astro, filename=fn)

    # if there is any filename the figure is closed after savefig
    # plt.show()
    assert fn.exists()
