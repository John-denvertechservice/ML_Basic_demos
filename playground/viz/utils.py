"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(train_losses, train_accuracies=None, val_losses=None, val_accuracies=None):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 2 if train_accuracies is not None else 1, figsize=(12, 4))
    if train_accuracies is None:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    if train_accuracies is not None:
        axes[1].plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
        if val_accuracies:
            axes[1].plot(epochs, val_accuracies, 'r-', label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    return plt


def plot_confusion_matrix_sklearn(y_true, y_pred, class_names=None):
    """Plot confusion matrix using sklearn metrics."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

