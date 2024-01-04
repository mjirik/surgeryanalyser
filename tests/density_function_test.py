from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.data
import skimage.io

from piglegcv import run_report


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
