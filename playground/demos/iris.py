"""Iris flower classification demo using scikit-learn."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from playground.utils.io import ensure_outdir, timestamped_path


def load_iris_data():
    """Load the Iris dataset."""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


def create_pipeline(classifier='svm'):
    """Create a sklearn pipeline with preprocessing and classifier."""
    scaler = StandardScaler()
    
    if classifier == 'svm':
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    elif classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', clf)
    ])
    
    return pipeline


def plot_feature_pairs(X, y, feature_names, target_names, save_plot=True):
    """Plot pairwise feature relationships."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    pair_idx = 0
    for i in range(4):
        for j in range(i+1, 4):
            ax = axes[pair_idx]
            scatter = ax.scatter(X[:, i], X[:, j], c=y, cmap='viridis', s=50, alpha=0.7)
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.set_title(f'{feature_names[i]} vs {feature_names[j]}')
            pair_idx += 1
    
    plt.tight_layout()
    return plt


def plot_confusion_matrix(y_true, y_pred, target_names, save_plot=True):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt


def run_iris_demo(classifier='svm', test_size=0.2, save_plots=True, outdir="outputs/plots"):
    """Run the Iris classification demo."""
    print("Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {target_names}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining {classifier.upper()} classifier...")
    pipeline = create_pipeline(classifier=classifier)
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining accuracy: {train_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    plt1 = plot_feature_pairs(X, y, feature_names, target_names, save_plot=save_plots)
    if save_plots:
        ensure_outdir(outdir)
        save_path1 = timestamped_path(outdir, stem="iris_feature_pairs", suffix=".png")
        plt1.savefig(save_path1, dpi=150, bbox_inches='tight')
        print(f"\nSaved feature pairs plot to {save_path1}")
    else:
        plt1.show()
    
    plt2 = plot_confusion_matrix(y_test, y_test_pred, target_names, save_plot=save_plots)
    if save_plots:
        ensure_outdir(outdir)
        save_path2 = timestamped_path(outdir, stem="iris_confusion_matrix", suffix=".png")
        plt2.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path2}")
    else:
        plt2.show()
    
    return pipeline, test_accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Iris flower classification demo')
    parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'rf'],
                        help='Classifier to use (svm or rf)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size fraction')
    parser.add_argument('--no-save', action='store_true',
                        help='Display plots instead of saving')
    parser.add_argument('--outdir', type=str, default='outputs/plots', help='Output directory for plots')
    args = parser.parse_args()
    
    run_iris_demo(classifier=args.classifier, test_size=args.test_size, save_plots=not args.no_save, outdir=args.outdir)

