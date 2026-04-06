"""
Utility functions for device management, random seeding, and logging.

Provides centralized setup for reproducibility and device handling across the project.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value

    Raises:
        ValueError: If seed is not a non-negative integer
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be non-negative integer, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def detect_device() -> torch.device:
    """
    Detect and return appropriate device (GPU or CPU).

    Returns:
        torch.device: GPU device if available, otherwise CPU

    Raises:
        RuntimeError: If CUDA initialization fails
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(
                f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB) | CUDA {torch.version.cuda}"
            )
            return device
    except RuntimeError as e:
        logger.warning(f"CUDA error during detection: {e}")

    device = torch.device("cpu")
    logger.info(f"Using CPU device")
    return device


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get device from string specification.

    Args:
        device_str: Device string ('auto', 'cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        torch.device: Requested device

    Raises:
        ValueError: If device string is invalid
    """
    device_str = device_str.lower().strip()

    if device_str == "auto":
        return detect_device()

    try:
        device = torch.device(device_str)
    except RuntimeError as e:
        raise ValueError(f"Invalid device string '{device_str}': {e}")

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    return device


def log_environment_info() -> None:
    """Log system, library, and hardware information at startup."""
    logger.info("=" * 70)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("=" * 70)

    # Python & system
    import sys

    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Platform: {sys.platform}")

    # PyTorch
    logger.info(f"PyTorch: {torch.__version__}")

    # PyG
    try:
        import torch_geometric

        logger.info(f"PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        logger.warning("PyTorch Geometric not installed")

    # NumPy, Pandas
    logger.info(f"NumPy: {np.__version__}")
    try:
        import pandas

        logger.info(f"Pandas: {pandas.__version__}")
    except ImportError:
        pass

    # Device
    device = detect_device()
    logger.info(f"Device: {device}")

    logger.info("=" * 70)


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path: Directory path (pathlib.Path object)

    Raises:
        OSError: If directory creation fails
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    Configure loguru logging with file and console output.

    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Raises:
        ValueError: If log level is invalid
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level '{level}'. Must be one of {valid_levels}")

    log_dir = ensure_dir(log_dir)

    # Remove default handler
    logger.remove()

    # Console handler — safe for Windows cp1252 and wandb stdout capture
    logger.add(
        lambda msg: print(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'), end=""),
        format="{level: <8} | {name}:{function} - {message}",
        level=level.upper(),
        colorize=False,
    )

    # File handler
    log_file = log_dir / f"training.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | {name}:{function}:{line} - {message}",
        level=level.upper(),
        rotation="100 MB",
        retention="10 days",
    )

    logger.info(f"Logging configured | Output: {log_file}")


def create_run_dir(base_dir: str = "runs") -> Path:
    """
    Create timestamped run directory.

    Args:
        base_dir: Base directory for runs

    Returns:
        Path: Created run directory

    Raises:
        OSError: If directory creation fails
    """
    from datetime import datetime

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(base_dir) / run_name)
    logger.info(f"Created run directory: {run_dir}")
    return run_dir
