from pathlib import Path
from typing import Union
import json
import os
from loguru import logger


def save_json(data:dict, output_json:Union[str,Path], update:bool=True):
    logger.debug(f"Writing '{output_json}'")

    output_json = Path(output_json)
    output_json.parent.mkdir(exist_ok=True, parents=True)
    # os.makedirs(os.path.dirname(output_json), exist_ok=True)
    dct = {}
    if update and output_json.exists():
        with open(output_json, "r") as output_file:
            dct = json.load(output_file)
        logger.debug(f"old keys: {list(dct.keys())}")
    dct.update(data)
    logger.debug(f"updated keys: {list(dct.keys())}")
    with open(output_json, "w") as output_file:
        json.dump(dct, output_file, indent=4)


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


def unit_conversion(value, input_unit:str, output_unit:str):
    in_kvantif = input_unit[-2] if len(input_unit) > 1 else ""
    out_kvantif = output_unit[-2] if len(output_unit) > 1 else ""

    in_k = _unit_multiplier(in_kvantif)
    out_k = _unit_multiplier(out_kvantif)

    return value * in_k / out_k


def _unit_multiplier(kvantif:str):
    multiplier = None
    if len(kvantif) == 0:
        multiplier = 1
    elif kvantif == "u":
        multiplier = 1e-6
    elif kvantif == "m":
        multiplier = 1e-3
    elif kvantif == "c":
        multiplier = 1e-2
    elif kvantif == "k":
        multiplier = 1e+3
    elif kvantif == "M":
        multiplier = 1e+6
    elif kvantif == "G":
        multiplier = 1e+9
    else:
        raise ValueError(f"Unknown unit kvantifier {kvantif}")

    return multiplier