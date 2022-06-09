import skimage.io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



def test_density():
    nsamples = 100
    data1 = np.random.normal([10, 6], [5, 4], size=[nsamples, 2])
    data2 = np.random.normal([60, 20], [10, 7], size=[nsamples, 2])
    data = np.concatenate([data1, data2], axis=0)
    print(data.shape)

    plt.plot(data[:, 0], data[:, 1], "*")
    df = pd.DataFrame(data, columns=['x', 'y'])

    sns.kdeplot(data=df, x='x', y='y', fill=True)

    # plt.show()