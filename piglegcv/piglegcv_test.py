import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pigleg_cv
import pytest
from skimage.io import imread


def test_empty_test():
    assert True


# def test_whole_piglegcv_test2():
#
#     do_the_test("*/pigleg_test2_2.mp4")

# def test_whole_piglegcv_test_video_end():
#     do_the_test("*/test_video_end.mp4")

# def test_whole_piglegcv_Anasto.mp4
#     do_the_test("*/Anasto*.mp4")

# @pytest.fixtures.paramteres([
#     "*/pigleg_test2_2.mp4",
#     "*/test_video_end.mp4",
#     "*/test_video_end.mp4"
# ])
@pytest.mark.parametrize(
    "path_mask,expected",
    [
        ("*/pigleg_test2_2.mp4", []),
        ("*/test_video_end.mp4", []),
        ("*/test_Einzelknopfnaht.mov", []),
        ("*/test_2023.02.22-B-Balßuweit-Kevin-Fortlaufend-transcutan_1.webm", [])
        # ("*/Anasto*.mp4", []),
    ],
)
def test_do_the_test(path_mask, expected):
    #     pigleg_test2
    img_pths = list(Path("../piglegsurgeryweb/media/upload/").glob(path_mask))
    img_pth = img_pths[0]

    outputdir = f"./del_pytest_video_output/{img_pth.stem}"
    outputdirp = Path(outputdir)
    if outputdirp.exists():
        shutil.rmtree(outputdirp)

    assert not outputdirp.exists()

    pigleg_cv.do_computer_vision(img_pth, outputdir, meta=None)

    assert outputdirp.exists()

    with open(outputdirp / "piglegcv_log.txt") as f:
        if "ERROR" in f.read():
            assert False, "Error in piglegcv_log.txt"
