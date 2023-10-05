import pytest
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
import pigleg_cv
import shutil

def test_empty_test():
    assert True

    
def test_whole_piglegcv():
    
    img_pths = list(Path("../piglegsurgeryweb/media/upload/").glob("*/Anasto*.mp4"))
    
    outputdir = "./del_pytest_video_output/"
    outputdirp = Path(outputdir)
    if outputdirp.exists():
        shutil.rmtree(outputdirp)
        
    assert not outputdirp.exists()
        
    
    pigleg_cv.do_computer_vision(img_pths[0], outputdir , meta=None)
    
    assert outputdirp.exists()
    
    with open(outputdirp / 'piglegcv_log.txt') as f:
        if 'ERROR' in f.read():
            assert False, "Error in piglegcv_log.txt"
    