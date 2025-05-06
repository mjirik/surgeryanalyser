import logging
import time
import traceback
from typing import Optional, Union

import psutil
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_torch_cuda_device_if_available(device: Union[None, int, str, torch.device] = 0) -> torch.device:
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
        elif device == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(f"Unknown device string: {device}. Falling back to CUDA or CPU.")
            device = 0  # Default fallback
    elif isinstance(device, int):
        pass
    elif isinstance(device, torch.device):
        return device
    else:
        logger.warning(f"Unknown device string: {device}. Falling back to CUDA or CPU.")
        device = 0

    # Handle torch device assignment
    if torch.cuda.is_available():
        new_device = torch.device(device)
    else:
        new_device = torch.device("cpu")
    logger.debug(f"new_device: {new_device}")
    print(f"new_device: {new_device}")
    return new_device


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


def wait_for_gpu_memory(required_memory_gb: float = 1.0, device: Union[int, str] = 0):
    """Wait until GPU memory is below threshold."""
    device = get_torch_cuda_device_if_available(device)

    # check if device is cpu
    if device.type == "cpu":
        logger.debug("No need to wait for CPU")
        return

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
        time.sleep(5)

def empty_cache_and_syncronize(device: Union[int, str] = 0):
    """Empty GPU cache and synchronize."""
    device = get_torch_cuda_device_if_available(device)
    if device.type == "cpu":
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
