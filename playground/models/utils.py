"""Model utilities and helpers."""

import torch
import torch.nn as nn


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, filepath):
    """Save a PyTorch model."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cpu'):
    """Load a PyTorch model."""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


def initialize_weights(m):
    """Initialize weights for neural network layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

