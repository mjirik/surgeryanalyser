from pathlib import Path
from typing import Optional, Union
import json
from loguru import logger
import numpy as np
import os

# Function to handle serialization
def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (set, complex)):
        return list(obj)  # Convert sets and complex numbers to lists
    elif isinstance(obj, bytes):
        return obj.decode()  # Convert bytes to string
    else:
        return str(obj)  # Convert other non-serializable types to string


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(data: dict, output_json: Union[str, Path], update: bool = True):
    logger.debug(f"Writing '{output_json}'")

    output_json = Path(output_json)
    output_json.parent.mkdir(exist_ok=True, parents=True)
    # os.makedirs(os.path.dirname(output_json), exist_ok=True)
    dct = {}
    if update and output_json.exists():
        try:
            with open(output_json, "r") as output_file:
                dct = json.load(output_file)
            logger.debug(f"old keys: {list(dct.keys())}")
        except Exception as e:
            import traceback
            backup_fn = output_json.with_suffix(".bak.json")
            backup_fn.unlink(minimal=True)
            output_json.rename(backup_fn)
            logger.error(traceback.format_exc())
            logger.error(f"JSON {output_json} is corrupted. Making backup and creating new one.")
    dct.update(data)
    logger.debug(f"updated keys: {list(dct.keys())}")
    with open(output_json, "w") as output_file:
        try:
            json.dump(dct, output_file, indent=4,
                      # cls=NumpyEncoder,  # here is necessary to solve all types of objects
                      default=serialize  # here we are solving only the non serializable objects
                      )

        except Exception as e:
            logger.error(f"Error writing json file {output_json}: {e}")
            logger.error(f"Data: {dct}")
            print_nested_dict_with_types(dct, 4)

            raise e


def load_json(filename: Union[str, Path]):
    filename = Path(filename)
    if os.path.isfile(filename):
        with open(filename, "r") as fr:
            try:
                data = json.load(fr)
            except ValueError as e:
                return {}
            return data
    else:
        return {}

def print_nested_dict_with_types(d: dict, indent: int = 0):
    for k, v in d.items():
        if isinstance(v, dict):
            logger.debug(f"{' ' * indent}{k}:")
            print_nested_dict_with_types(v, indent + 2)
        else:
            logger.debug(f"{' ' * indent}{k}: {type(v)}")
