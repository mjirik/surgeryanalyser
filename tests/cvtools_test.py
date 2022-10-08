import pytest
from piglegcv import tools
from pathlib import Path



def test_save_json():
    fn =  Path("test_file.json")
    data1 = {"a": 1, "b": 2, "d": None}
    data2 = {"a": 3, "c": 4}
    expected_data = {"a": 3, "b": 2, "c": 4, "d": None}
    tools.save_json(data1, fn)
    tools.save_json(data2, fn)


    data = tools.load_json(fn)
    assert data == expected_data
