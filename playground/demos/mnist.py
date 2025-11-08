"""MNIST digit recognition demo using PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from playground.utils.io import ensure_outdir, timestamped_path


class MNISTNet(nn.Module):
    """Lightweight MLP for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def load_mnist_data(root='./data', train=True, subset_size=None):
    """Load MNIST dataset, optionally using a subset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    
    if subset_size and subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
    
    return dataset


def train_model(model, train_loader, epochs=3, lr=0.001):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


def evaluate_model(model, test_loader):
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def visualize_predictions(model, test_loader, num_samples=8):
    """Visualize some predictions."""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        outputs = model(images[:num_samples])
        predictions = outputs.argmax(dim=1)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {labels[i].item()}, Pred: {predictions[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return plt


def run_mnist_demo(epochs=3, quick=False, batch_size=64, save_plot=True, outdir="outputs/plots"):
    """Run the MNIST digit recognition demo."""
    train_subset_size = 1000 if quick else None
    test_subset_size = 200 if quick else None
    
    print("Loading MNIST dataset...")
    train_dataset = load_mnist_data(train=True, subset_size=train_subset_size)
    test_dataset = load_mnist_data(train=False, subset_size=test_subset_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cpu')
    model = MNISTNet(input_size=784, hidden_size=128, num_classes=10).to(device)
    
    print(f"Training model for {epochs} epochs...")
    train_model(model, train_loader, epochs=epochs)
    
    print("Evaluating on test set...")
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    plt_obj = visualize_predictions(model, test_loader)
    if save_plot:
        ensure_outdir(outdir)
        save_path = timestamped_path(outdir, stem="mnist_predictions", suffix=".png")
        plt_obj.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt_obj.show()
    
    return model, test_accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MNIST digit recognition demo')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--quick', action='store_true', help='Use smaller dataset for faster execution')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--no-save', action='store_true', help='Display plot instead of saving')
    parser.add_argument('--outdir', type=str, default='outputs/plots', help='Output directory for plots')
    args = parser.parse_args()
    
    run_mnist_demo(epochs=args.epochs, quick=args.quick, batch_size=args.batch_size, save_plot=not args.no_save, outdir=args.outdir)

