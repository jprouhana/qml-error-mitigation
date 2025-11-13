"""
Visualization functions for error mitigation experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_accuracy_vs_noise(results_dict, ideal_acc=None, save_dir='results'):
    """Line plot: accuracy vs error rate for different methods."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'No mitigation': '#FF6B6B', 'ZNE': '#4ECDC4',
              'Measurement': '#45B7D1', 'Combined': '#9B59B6'}

    for name, data in results_dict.items():
        ax.plot(data['error_rates'], data['accuracies'], 'o-',
                color=colors.get(name, 'gray'), linewidth=2, markersize=7, label=name)

    if ideal_acc is not None:
        ax.axhline(y=ideal_acc, color='black', linestyle='--', linewidth=1.5,
                   label=f'Ideal ({ideal_acc:.2f})', alpha=0.7)

    ax.set_xlabel('Depolarizing Error Rate')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Error Mitigation Comparison for VQC')
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'mitigation_comparison.png', dpi=150)
    plt.close()


def plot_zne_extrapolation(stretch_factors, values, extrapolated, save_dir='results'):
    """Show ZNE extrapolation curve."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stretch_factors, values, 'o', color='#FF6B6B', markersize=10, label='Measured')

    # fit line
    x_fit = np.linspace(0, max(stretch_factors), 100)
    coeffs = np.polyfit(stretch_factors, values, 1)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='gray', alpha=0.7)
    ax.plot(0, extrapolated, 's', color='#4ECDC4', markersize=12, label='Extrapolated')

    ax.set_xlabel('Noise Stretch Factor')
    ax.set_ylabel('Expectation Value')
    ax.set_title('Zero-Noise Extrapolation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'zne_extrapolation.png', dpi=150)
    plt.close()


def plot_confusion_matrices(results_dict, save_dir='results'):
    """Grid of confusion matrices for different methods."""
    pass  # implemented inline in notebook


def plot_decision_boundaries(models_dict, X, y, save_dir='results'):
    """Side-by-side decision boundaries."""
    pass  # implemented inline in notebook
