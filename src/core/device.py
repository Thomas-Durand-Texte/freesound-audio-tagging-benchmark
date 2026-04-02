"""Device management utilities for PyTorch operations on M4 MacBook Air.

This module centralizes all CPU/GPU device selection logic, providing utilities
to leverage Apple Silicon MPS (Metal Performance Shaders) acceleration when available.
"""

import torch


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations.

    Checks for MPS (Metal Performance Shaders) availability on Apple Silicon.
    Falls back to CPU if MPS is not available or not properly built.

    Returns:
        torch.device: MPS device if available and built, otherwise CPU device.
    """
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("Warning: MPS not built properly, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")


def print_device_info() -> None:
    """Print detailed information about available compute devices.

    Displays MPS availability, build status, and PyTorch version for debugging.
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    device = get_device()
    print(f"Using device: {device}")


def get_mps_memory_allocated() -> float:
    """Get current MPS memory allocation in MB.

    Returns:
        float: Memory allocated in megabytes. Returns 0.0 if MPS is not available.
    """
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0.0


def clear_mps_cache() -> None:
    """Clear MPS memory cache to free up memory.

    Call this when you need to free memory between training runs or experiments.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
