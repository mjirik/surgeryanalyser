import shutil

from loguru import logger
from pathlib import Path
# from unittest import mock
from piglegcv import pigleg_cv
# from matplotlib import pyplot as plt

local_dir = Path(__file__).parent

# from unittest.mock import patch
# @patch("torch.config")
def test_pigleg_cv():
    # mock.patch
    logger.debug("test init")
    # import piglegsurgeryweb

    media_pth = local_dir / "pigleg_test.mp4"
    assert media_pth.exists()
    outputdir = Path("test_pigleg_cv_output")
    # remove dir
    if outputdir.exists():
        shutil.rmtree(outputdir)
    # outputdir.mkdir(parents=True, exist_ok=True)
    #
    pigleg_cv.do_computer_vision(media_pth, outputdir)
    # pigleg_cv.run_media_processing(media_pth, outputdir)
    assert len(list(outputdir.glob("*"))) > 0
    assert False


