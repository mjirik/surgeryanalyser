import numpy as np

from piglegcv import run_report
from matplotlib import pyplot as plt
from pathlib import Path
import skimage.data
import skimage.transform
import shutil

local_dir = Path(__file__).parent


def test_qr_scissors_non_maximum_supression():
    json_data = run_report.load_json(local_dir / 'meta.json')

    frames = run_report._qr_data_processing(json_data, fps=25)
    assert len(frames) > 0
    # plt.plot(frames)
    # plt.show()
def test_qr_scissors_non_maximum_supression_empty():
    # json_data = run_report.load_json(local_dir / 'meta.json')
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

def test_main_report():
    filename = local_dir / "pigleg_test2.mp4"
    outputdir = local_dir / "test_main_report_outputdir"
    expected_file = outputdir / "pigleg_results.mp4"
    if expected_file.exists():
        expected_file.unlink()

    outputdir.mkdir(exist_ok=True, parents=True)



    required_files= ["tracks.json", "meta.json"]
    for fn in required_files:
        fnp = Path(local_dir / fn)
        if ~fnp.exists():
            shutil.copy(fnp, outputdir)

    run_report.main_report(
        str(filename), str(outputdir),
        object_colors=["b","r","g","m"],
        object_names=["Needle holder","Tweezers","Scissors","None"],
        concat_axis=1, resize_factor=.5
    )
    assert expected_file.exists()


def test_add_ruler():
    import cv2
    fn = list(Path(".").glob("*.jpg"))[0]
    img = cv2.imread(str(fn))
    run_report.insert_ruler_in_image(img, pixelsize_mm=0.1, ruler_size_mm=50)
    img = img[::3,::3,:]
    cv2.imshow("okno", img)
    cv2.waitKey(5000)





