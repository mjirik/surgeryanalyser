import pytest
import pythontemplate.moduleone
from pathlib import Path
rootdir = Path(__file__).parent.parent
import sys
# pigleg_pth = rootdir/ "piglegsurgeryweb"
# sys.path.insert(0, str(pigleg_pth))
from loguru import logger


def test_hello():
    pythontemplate.moduleone.print_hello("Vlkoslav")

def test_pigleg_cv():
    logger.debug("test init")
    import piglegsurgeryweb
    from piglegsurgeryweb.uploader import pigleg_cv

    media_pth = Path(r"H:\biomedical\orig\pigleg_surgery\first_dataset\b6c6fb92-d8ad-4ccf-994c-5241a89a9274.mp4")
    outputdir = Path("test_pigleg_cv_output")
    pigleg_cv.run_media_processing(media_pth, outputdir)