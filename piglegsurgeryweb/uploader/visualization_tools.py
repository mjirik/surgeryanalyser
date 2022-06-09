import numpy as np


def crop_square(frame:np.ndarray)->np.ndarray:

    mn = np.min(frame.shape[:2])
    sh0 = frame.shape[0]
    sh1 = frame.shape[1]
    if sh0 > sh1:
        st0 = int((sh0/2) - (sh1/2))
        st1 = 0
    else:
        st0 = 0
        st1 = int((sh1/2) - (sh0/2))

    frame = frame[st0:st0+mn, st1:st1+mn]

    return frame
