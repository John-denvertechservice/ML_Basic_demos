"""CLI entry point for Tiny ML Playground demos."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Tiny ML Playground - CPU-friendly ML experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available demos:
  spirals  - Synthetic spiral classification using PyTorch (no downloads)
  mnist    - MNIST digit recognition using PyTorch (~60MB download)
  iris     - Iris flower classification using scikit-learn (no downloads)

Examples:
  python -m playground.cli spirals --epochs 5
  python -m playground.cli mnist --quick
  python -m playground.cli iris --classifier rf
        """
    )
    
    parser.add_argument(
        'demo',
        choices=['spirals', 'mnist', 'iris'],
        help='Demo to run'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use smaller datasets for faster execution'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Display plots instead of saving to files'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='outputs/plots',
        help='Output directory for plots (default: outputs/plots)'
    )
    
    args, unknown = parser.parse_known_args()
    
    if args.demo == 'spirals':
        from playground.demos.spirals import run_spirals_demo
        run_spirals_demo(
            epochs=args.epochs,
            quick=args.quick,
            save_plot=not args.no_save,
            outdir=args.outdir
        )
    
    elif args.demo == 'mnist':
        from playground.demos.mnist import run_mnist_demo
        run_mnist_demo(
            epochs=args.epochs,
            quick=args.quick,
            save_plot=not args.no_save,
            outdir=args.outdir
        )
    
    elif args.demo == 'iris':
        from playground.demos.iris import run_iris_demo
        iris_parser = argparse.ArgumentParser()
        iris_parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'rf'])
        iris_parser.add_argument('--test-size', type=float, default=0.2)
        iris_args, _ = iris_parser.parse_known_args(unknown)
        
        run_iris_demo(
            classifier=iris_args.classifier,
            test_size=iris_args.test_size,
            save_plots=not args.no_save,
            outdir=args.outdir
        )


if __name__ == '__main__':
    main()

