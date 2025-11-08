# Tiny ML Playground

CPU-friendly ML experiments optimized for 2019 Intel MacBook Pro.

## Quick Start

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run a demo:**
   ```bash
   python -m playground.cli spirals
   python -m playground.cli mnist --quick
   python -m playground.cli iris --classifier rf
   ```

## Available Demos

- **spirals** - Synthetic spiral classification using PyTorch (no downloads)
- **mnist** - MNIST digit recognition using PyTorch (~60MB download)
- **iris** - Iris flower classification using scikit-learn (no downloads)

## Output Files

All generated plots and figures are saved to the `outputs/` directory by default:

- `outputs/plots/` - PNG files and visualizations (default location)
- `outputs/models/` - Model checkpoints (if any)
- `outputs/logs/` - Run metadata (optional)

Files are automatically timestamped to avoid overwriting previous runs. You can specify a custom output directory using the `--outdir` flag:

```bash
python -m playground.cli spirals --outdir my_outputs/plots
```

## CLI Options

Common flags available for all demos:
- `--epochs N` - Number of training epochs (default: 3)
- `--quick` - Use smaller datasets for faster execution
- `--no-save` - Display plots instead of saving to files
- `--outdir PATH` - Output directory for plots (default: outputs/plots)

## Project Structure

```
playground/
  cli.py          # Main CLI entry point
  demos/          # Demo implementations
  data/           # Data loading utilities
  models/          # Model definitions
  viz/            # Visualization helpers
  utils/          # Utility functions (I/O, etc.)
notebooks/        # Jupyter notebooks
outputs/          # Generated outputs
  plots/          # PNG files and figures
  models/         # Model checkpoints
  logs/           # Run metadata
```

## Requirements

See `requirements.txt` for pinned dependencies. All packages are CPU-only builds optimized for Intel processors.

