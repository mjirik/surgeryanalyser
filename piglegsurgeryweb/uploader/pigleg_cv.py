from pathlib import Path


def run_media_processing(filename: Path, outputdir: Path) -> dict:
    """
    Based on filename suffix the processing
    :param filename:
    :param outputdir:
    :return:
    """
    isvideo = True
    if isvideo:
        return run_video_processing(filename, outputdir)
    else:
        return run_image_processing(filename, outputdir)


def run_video_processing(filename: Path, outputdir: Path) -> dict:
    # TODO here will be tracking
    return {
        "Needle Holder Tip Track Length [m]": 123.5,
        "Needle Holder Tip Avg Velocity [ms^1]": 123.5,
    }


def run_image_processing(filename: Path, outputdir: Path) -> dict:
    # TODO here will be angle measurement
    return {
        "Stitch Angle 1 [°]": 0.75,
        "Stitch Angle 2 [°]": 0.75,
    }
