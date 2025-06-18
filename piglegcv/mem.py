import logging
import time
import traceback
from typing import Optional, Union
from pathlib import Path

import psutil
import torch

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
from loguru import logger


LOADED_MODELS = {}


def load_model_torch(model_path: Union[str, Path], required_memory_gb: float = 1.0, device: Union[int, str, torch.device] = 0) -> torch.nn.Module:
    """Load a PyTorch model from a file."""
    global LOADED_MODELS
    device = get_torch_cuda_device_if_available(device)
    key = str(model_path) + "_" + str(device)
    logger.debug("Loading model {}".format(key))
    if key in LOADED_MODELS:
        logger.debug(f"Model {model_path} already loaded on device {device}.")
        print(f"Model {model_path} already loaded on device {device}.")
        return LOADED_MODELS[key]

    wait_for_gpu_memory(required_memory_gb, device=device)

    model = torch.load(model_path, map_location=device)

    LOADED_MODELS[key] = model
    return model

def release_model_torch(model_path: Union[str, Path], device: Union[int, str, torch.device] = 0) -> dict:
    """Delete a loaded PyTorch model."""
    global LOADED_MODELS
    key = str(model_path) + "_" + str(device)

    if model_path in LOADED_MODELS:
        del LOADED_MODELS[key]
        empty_cache_and_syncronize(device= device)
        logger.debug(f"Deleted model {model_path} from memory.")
    else:
        logger.warning(f"Model {model_path} not found in loaded models.")
    return LOADED_MODELS


def get_torch_cuda_device_if_available(device: Union[None, int, str, torch.device] = 0, allow_cpu=True) -> torch.device:
    """Set and return a valid torch device."""
    logger.debug(f"requested device: {device}")
    print(f"requested device: {device}")

    if isinstance(device, str):
        # Handle string input like 'cuda', 'cuda:0', or 'cpu'
        if device.startswith("cuda"):
            if ":" not in device:
                device = 0  # Default to cuda:0 if no index is provided
            else:
                device = int(device.split(":")[-1])  # Extract index
            device = torch.device(device)
        elif device == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(f"Unknown device string: {device}. Falling back to CUDA or CPU.")
            # device = 0  # Default fallback
            device = torch.device(0)
    elif isinstance(device, int):
        pass
    elif isinstance(device, torch.device):
        if device.type == "cpu":
            return device
        else:
            pass
    else:
        logger.warning(f"Unknown device string: {device}. Falling back to CUDA or CPU.")
        device = torch.device(0)

    # Handle torch device assignment
    # new_device = wait_for_gpu_device(device, max_wait_time_s=1800, allow_cpu=allow_cpu)
    new_device = device
    logger.debug(f"new_device: {new_device}")
    print(f"new_device: {new_device}")
    return new_device


def wait_for_gpu_device(device: torch.device, max_wait_time_s: int = 3600, allow_cpu: bool = True):
    """Wait until a GPU device is available."""
    start_time = time.time()
    while True:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return device
        else:
            logger.debug("Waiting for GPU device to become available.")
            print("Waiting for GPU device to become available.")

        if time.time() - start_time > max_wait_time_s:
            logger.debug(f"Timeout ({max_wait_time_s} [s]) waiting for GPU device {device}.")
            print(f"Timeout waiting for GPU device {device}.")
            if allow_cpu:
                logger.debug("Falling back to CPU.")
                print("Falling back to CPU.")
                return torch.device("cpu")
            else:
                logger.error(f"GPU device {device} not available after {max_wait_time_s} seconds.")
                print(f"GPU device {device} not available after {max_wait_time_s} seconds.")
                return None
        time.sleep(5)



def get_ram():
    """Get visualized RAM usage in GB."""
    mem = psutil.virtual_memory()
    free = mem.available / 1024**3
    total = mem.total / 1024**3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return (
        f"RAM:  {total - free:.1f}/{total:.1f}GB  RAM: ["
        + (total_cubes - free_cubes) * "▮"
        + free_cubes * "▯"
        + "]"
    )


def get_vram(device: Optional[torch.device] = None) -> str:
    """Get visualized VRAM usage in GB."""
    device = get_torch_cuda_device_if_available(device)
    device = device if device else torch.cuda.current_device()
    if torch.device(device).type == "cpu":
        return "No GPU available"
    try:
        free = torch.cuda.mem_get_info(device)[0] / 1024**3
        total = torch.cuda.mem_get_info(device)[1] / 1024**3
        used = total - free
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return (
            f"device:{device}    VRAM: {total - free:.1f}/{total:.1f}GB  VRAM:["
            + (total_cubes - free_cubes) * "▮"
            + free_cubes * "▯"
            + "]"
        )
    except ValueError:
        logger.debug(f"device: {device}, {torch.cuda.is_available()=}")
        logger.error(f"Error: {traceback.format_exc()}")
        return "No GPU available"


def wait_for_gpu_memory(required_memory_gb: float = 1.0, device: Union[int, str, torch.device] = 0, max_wait_time_s: int = 3600):
    """Wait until GPU memory is below threshold."""
    logger.debug(f"Waiting for {required_memory_gb} GB of GPU memory on device {device}.")
    print(f"Waiting for {required_memory_gb} GB of GPU memory on device {device}.")
    device = get_torch_cuda_device_if_available(device)
    logger.debug(f"device: {device}")

    # check if device is cpu
    if device.type == "cpu":
        logger.debug("No need to wait for CPU")
        return

    start_time = time.time()
    while True:
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        free_memory_gb = total - reserved
        if free_memory_gb > required_memory_gb:
            logger.debug(f"Free memory: {free_memory_gb:.1f} GB > {required_memory_gb} GB")
            print(f"Free memory: {free_memory_gb:.1f} GB > {required_memory_gb} GB")
            break
        logger.debug(f"Waiting for {required_memory_gb} GB of GPU memory. " + get_vram(device))
        print(f"Waiting for {required_memory_gb} GB of GPU memory. " + get_vram(device))
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if time.time() - start_time > max_wait_time_s:
            logger.debug(f"Timeout ({max_wait_time_s} [s]) waiting for GPU memory. Free memory: {free_memory_gb:.1f} GB")
            print(f"Timeout waiting for GPU memory. Free memory: {free_memory_gb:.1f} GB")
            break
        time.sleep(5)

def empty_cache_and_syncronize(device: Union[int, str] = 0):
    """Empty GPU cache and synchronize."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    device = get_torch_cuda_device_if_available(device)
    logger.debug(f"device={device}")
    # just print where from is the function called
    if device.type == "cpu":
        return
