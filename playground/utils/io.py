"""I/O utilities for file management."""

from pathlib import Path
from datetime import datetime


def ensure_outdir(outdir: str) -> Path:
    """Ensure output directory exists, creating it if necessary."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    return outdir_path


def timestamped_path(outdir: str, stem: str, suffix: str = ".png") -> Path:
    """Generate a timestamped file path in the output directory.
    
    Args:
        outdir: Output directory path
        stem: Base filename (without extension)
        suffix: File extension (default: ".png")
    
    Returns:
        Path object with timestamped filename
    """
    ensure_outdir(outdir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}_{timestamp}{suffix}"
    return Path(outdir) / filename

