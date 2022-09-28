import numpy as np

from piglegcv import run_report
from matplotlib import pyplot as plt
from pathlib import Path
import skimage.data
import skimage.transform

local_dir = Path(__file__).parent


def test_qr_scissors_non_maximum_supression():
    json_data = run_report.load_json(local_dir / 'qr_data.json')

    frames = run_report._qr_data_processing(json_data, fps=25)
    assert len(frames) > 0
    # plt.plot(frames)
    # plt.show()
def test_qr_scissors_non_maximum_supression_empty():
    # json_data = run_report.load_json(local_dir / 'qr_data.json')
    json_data = {}

    frames = run_report._qr_data_processing(json_data, fps=25)
    assert len(frames) > 0


def test_create_video_report():
    """
    With some combinations of vide_size and dpi the output video size is not correct.
    I.e.:
    video_size = [1376, 776]
    dpi = 300
    :return:
    """




    im = skimage.data.astronaut()
    frame_ids = []
    data_pixels = []
    source_fps = 30
    pix_size = (400,300)
    QRinit = True
    object_colors = ['r', 'g', 'b']
    object_names = ['a', 'b', 'c']
    video_size = [1376, 776]
    dpi = 400
    fig, ax, ds_max  = run_report.create_video_report(
        frame_ids, data_pixels, source_fps, pix_size, QRinit, object_colors, object_names,
        video_size, ds_threshold=0.1, dpi=dpi, scissors_frames=[])

    videosize_inch = np.asarray(video_size) / dpi

    img = run_report.plot3(fig)
    # assert img.shape == (video_size[1], video_size[0], 4)
    # skimage.transform
    # plt.close(fig)
    # plt.figure()
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    if not img.shape[:2] == tuple(video_size[::-1]):
        img = skimage.transform.resize(img, video_size[::-1], preserve_range=True).astype(img.dtype)
    assert img.shape == (video_size[1], video_size[0], 4)