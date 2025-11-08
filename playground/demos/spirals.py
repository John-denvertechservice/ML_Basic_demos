"""Synthetic spirals classification demo using PyTorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from playground.utils.io import ensure_outdir, timestamped_path


def generate_spirals(n_samples=1000, n_classes=3, noise=0.1):
    """Generate synthetic spiral dataset."""
    X = []
    y = []
    
    for class_id in range(n_classes):
        t = np.linspace(0, 4 * np.pi, n_samples // n_classes)
        r = t / (4 * np.pi)
        
        for i in range(len(t)):
            x1 = r[i] * np.cos(t[i] + 2 * np.pi * class_id / n_classes) + np.random.normal(0, noise)
            x2 = r[i] * np.sin(t[i] + 2 * np.pi * class_id / n_classes) + np.random.normal(0, noise)
            X.append([x1, x2])
            y.append(class_id)
    
    return np.array(X), np.array(y)


class SpiralNet(nn.Module):
    """Simple neural network for spiral classification."""
    
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=3):
        super(SpiralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, epochs=3, lr=0.01):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def plot_decision_boundary(model, X, y, resolution=100):
    """Plot decision boundary and data points."""
    device = next(model.parameters()).device
    model.eval()
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = torch.argmax(Z, dim=1).cpu().numpy()
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'][:len(np.unique(y))])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='black', s=20)
    plt.colorbar(scatter)
    plt.title('Spiral Classification - Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.tight_layout()
    return plt


def run_spirals_demo(epochs=3, n_samples=1000, quick=False, save_plot=True, outdir="outputs/plots"):
    """Run the spirals classification demo."""
    if quick:
        n_samples = 300
    
    print(f"Generating {n_samples} spiral samples...")
    X, y = generate_spirals(n_samples=n_samples, n_classes=3)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cpu')
    model = SpiralNet(input_dim=2, hidden_dim=64, num_classes=3).to(device)
    
    print(f"Training model for {epochs} epochs...")
    train_model(model, train_loader, epochs=epochs)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()
    
    print(f"Training accuracy: {accuracy*100:.2f}%")
    
    plt_obj = plot_decision_boundary(model, X, y)
    if save_plot:
        ensure_outdir(outdir)
        save_path = timestamped_path(outdir, stem="spirals_decision_boundary", suffix=".png")
        plt_obj.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt_obj.show()
    
    return model, accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Synthetic spirals classification demo')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--quick', action='store_true', help='Use smaller dataset for faster execution')
    parser.add_argument('--no-save', action='store_true', help='Display plot instead of saving')
    parser.add_argument('--outdir', type=str, default='outputs/plots', help='Output directory for plots')
    args = parser.parse_args()
    
    run_spirals_demo(epochs=args.epochs, quick=args.quick, save_plot=not args.no_save, outdir=args.outdir)

