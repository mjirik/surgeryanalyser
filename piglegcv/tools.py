from pathlib import Path
from typing import Union
import json
import os
from loguru import logger


def save_json(data:dict, output_json:Union[str,Path]):
    logger.debug(f"Writing '{output_json}'")

    output_json = Path(output_json)
    output_json.parent.mkdir(exist_ok=True, parents=True)
    # os.makedirs(os.path.dirname(output_json), exist_ok=True)
    dct = {}
    if output_json.exists():
        with open(output_json, "r") as output_file:
            dct = json.load(output_file)

    dct.update(data)
    with open(output_json, "w") as output_file:
        json.dump(dct, output_file)


def load_json(filename:Union[str,Path]):
    filename = Path(filename)
    if os.path.isfile(filename):
        with open(filename, 'r') as fr:
            try:
                data = json.load(fr)
            except ValueError as e:
                return {}
            return data
    else:
        return {}

