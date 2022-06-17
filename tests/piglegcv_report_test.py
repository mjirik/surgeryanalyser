from piglegcv import run_report
from matplotlib import pyplot as plt
from pathlib import Path

local_dir = Path(__file__).parent


def test_qr_scissors_non_maximum_supression():
    json_data = run_report.load_json(local_dir / 'qr_data.json')

    frames = run_report._qr_data_processing(json_data, fps=25)
    # plt.plot(frames)
    # plt.show()
