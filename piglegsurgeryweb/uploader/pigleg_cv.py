from pathlib import Path
import cv2
import json


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
    # outputdir = Path(outputdir)
    # filename = Path(str(filename))
    outputdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("tmp_video_processing") / filename.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(filename))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()

        frame_id += 1
        if not ret:
            break
        else:
            file_name = "{}/frame_{:0>6}.png".format(tmp_dir, frame_id)
            cv2.imwrite(file_name, frame)
    cap.release()

    json_file = tmp_dir / "meta.json"
    json.dump({}, json_file)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process media file")
    parser.add_argument("filename", type=str)
    parser.add_argument("outputdir", type=str)
    args = parser.parse_args()
    run_media_processing(Path(args.filename), Path(args.outputdir))
