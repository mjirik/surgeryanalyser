import numpy as np

try:
    from piglegcv import run_report
except ImportError:
    from .piglegcv import run_report

import shutil
from pathlib import Path

import cv2
import skimage.data
import skimage.transform
from matplotlib import pyplot as plt

local_dir = Path(__file__).parent


def test_qr_scissors_non_maximum_supression():
    json_data = run_report.load_json(local_dir / "meta.json")

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
    pix_size = (400, 300)
    QRinit = True
    object_colors = ["r", "g", "b"]
    object_names = ["a", "b", "c"]
    video_size = [1376, 776]
    dpi = 400
    fig, ax, ds_max, cumulative_measurements = run_report.create_video_report_figure(
        frame_ids,
        data_pixels,
        source_fps,
        pix_size,
        QRinit,
        object_colors,
        object_names,
        video_size,
        ds_threshold=0.1,
        dpi=dpi,
        cut_frames=[],
    )

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
        img = skimage.transform.resize(
            img, video_size[::-1], preserve_range=True
        ).astype(img.dtype)
    assert img.shape == (video_size[1], video_size[0], 4)


def unzip_file(zip_file, output_dir):
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def test_main_report():
    filename = local_dir / "pigleg_test2.mp4"
    outputdir = local_dir / "test_main_report_outputdir"
    expected_file = outputdir / "pigleg_results.mp4"
    if expected_file.exists():
        expected_file.unlink()

    outputdir.mkdir(exist_ok=True, parents=True)

    required_files = ["tracks.json", "meta.json"]
    for fn in required_files:
        fnp = Path(local_dir / fn)
        if ~fnp.exists():
            shutil.copy(fnp, outputdir)

    run_report.main_report(
        str(filename),
        str(outputdir),
        object_colors=["b", "r", "g", "m"],
        object_names=["Needle holder", "Tweezers", "Scissors", "None"],
        concat_axis=1,
    )
    assert expected_file.exists()


def test_main_report_micro():

    test_data_dir = (
        local_dir / "test_data/test_micro_SA_20230526-153002VID20230526111625.mp4"
    )
    assert (
        test_data_dir.exists()
    ), "Unpack test_micro_SA_20230526-15300...zip into test_data directory "

    # if not test_data_dir.exists():
    #     zip_path = list(local_dir / "test_data/").glob("test_micro*.zip")[0]
    #     assert zip_path.exists()
    #
    #
    # zip_output_dir = outputdir = local_dir / "test_main_report_outputdir"
    # shutil.rmtree(zip_output_dir, ignore_errors=True)
    # zip_output_dir.mkdir(exist_ok=True, parents=True)
    # unzip_file(zip_path, zip_output_dir)

    filename = test_data_dir / "__cropped.mp4"
    outputdir = local_dir / "test_main_report_outputdir"
    expected_file = outputdir / "pigleg_results.mp4"
    if expected_file.exists():
        expected_file.unlink()

    outputdir.mkdir(exist_ok=True, parents=True)

    required_files = ["tracks.json", "meta.json"]
    for fn in required_files:
        fnp = Path(test_data_dir / fn)
        if ~fnp.exists():
            shutil.copy(fnp, outputdir)

    meta = run_report.load_json(Path(test_data_dir / "meta.json"))

    run_report.main_report(
        str(filename),
        str(outputdir),
        is_microsurgery=True,
        # object_colors=["b", "r", "g", "m"],
        # object_names=["Needle holder", "Tweezers", "Scissors", "None"],
        concat_axis=1,
        meta=meta,
    )
    assert expected_file.exists()


def test_add_ruler_is_micro():
    test_data_dir = (
        local_dir / "test_data/test_micro_SA_20230526-153002VID20230526111625.mp4"
    )
    assert test_data_dir.exists()

    meta = run_report.load_json(test_data_dir / "meta.json")
    meta_qr = meta["qr_data"]
    # filename = str(filename)
    # outputdir = str(outputdir)
    filename = test_data_dir / "__cropped.mp4"

    cap = cv2.VideoCapture(str(filename))
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert cap.isOpened(), f"Failed to load video file {filename}"

    # output video
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    size_input_video = [
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    ]
    shape = size_input_video + [3]

    pix_size = meta_qr["pix_size"]
    ruler_size_unit = 10 if meta_qr["is_microsurgery"] else 50
    visualization_length_unit = "mm" if meta_qr["is_microsurgery"] else "cm"
    # logger.debug(f"{pixelsize=}, {ruler_size=}, {visualization_length_unit=}")

    ruler_adder = run_report.AddRulerInTheFrame(
        shape,
        pix_size_m=pix_size,
        ruler_size=ruler_size_unit,
        unit=visualization_length_unit,
    )

    assert np.sum(ruler_adder.mask) > 0
    # plt.imshow(ruler_adder.mask)
    # plt.show()

    flag, img = cap.read()
    img = ruler_adder.add_in_the_frame(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    assert ruler_adder.mask.shape[2] == img.shape[2]


def test_add_ruler():
    import cv2

    fn = list(Path(".").glob("*.jpg"))[0]
    img = cv2.imread(str(fn))
    run_report.insert_ruler_in_image(img, pixelsize=0.1, ruler_size=50)
    img = img[::3, ::3, :]
    plt.imshow(img[:, :, ::-1])
    plt.show()
    # cv2.imshow("okno", img)
    # cv2.waitKey(5000)
