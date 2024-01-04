from pathlib import Path

import numpy as np
import pytest

from piglegcv import tools


def test_save_json():
    fn = Path("test_file.json")
    data1 = {"a": 1, "b": 2, "d": None}
    data2 = {"a": 3, "c": 4}
    expected_data = {"a": 3, "b": 2, "c": 4, "d": None}
    tools.save_json(data1, fn)
    tools.save_json(data2, fn)

    data = tools.load_json(fn)
    assert data == expected_data


def test_unit_conversion():
    v = tools.unit_conversion(100, "cm", "mm")
    assert v == 1000

    v = tools.unit_conversion(np.asarray([1000, 1500]), "mm", "m")
    assert v[0] == 1.0
    assert v[1] == 1.5
