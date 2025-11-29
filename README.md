# QML Error Mitigation

Benchmarking quantum error mitigation techniques for variational quantum classifiers. Studies how noise degrades QML model accuracy and evaluates zero-noise extrapolation and measurement error mitigation as recovery strategies.

Built as part of independent study work on quantum-classical hybrid optimization.

## Background

### The Noise Problem for QML

Near-term quantum computers are noisy. Gate errors, decoherence, and measurement errors all degrade the performance of variational quantum algorithms. For quantum machine learning, this means a classifier that works perfectly on a noiseless simulator may fail completely on real hardware.

Understanding and mitigating these errors is essential for any practical QML deployment.

### Mitigation Strategies

This project implements and compares two complementary approaches:

**Zero-Noise Extrapolation (ZNE):** Run the same circuit at multiple noise levels (by intentionally amplifying noise via gate folding), then extrapolate the results to the zero-noise limit. The key insight is that if we can measure how a quantity changes with noise, we can estimate what it would be without noise.

Gate folding works by replacing each gate $G$ with $G \cdot G^\dagger \cdot G$ (stretch factor 3) or $G \cdot G^\dagger \cdot G \cdot G^\dagger \cdot G$ (stretch factor 5), which increases the effective noise while leaving the ideal computation unchanged.

**Measurement Error Mitigation:** Calibrate the readout errors by preparing known computational basis states and measuring them. This gives a calibration matrix $A$ where $A_{ij} = P(\text{measure } i \mid \text{prepared } j)$. The true probability distribution can then be recovered by inverting this matrix.

## Project Structure

```
qml-error-mitigation/
├── src/
│   ├── noise_models.py        # Noise model construction
│   ├── mitigation.py           # ZNE and measurement mitigation
│   ├── noisy_classifier.py     # VQC training under noise
│   ├── data_utils.py           # Dataset loading
│   └── plotting.py             # Visualization functions
├── notebooks/
│   └── error_mitigation_analysis.ipynb
├── results/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/qml-error-mitigation.git
cd qml-error-mitigation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.noise_models import build_depolarizing_model
from src.noisy_classifier import train_noisy_vqc, evaluate_classifier
from src.data_utils import load_moons_dataset

X_train, X_test, y_train, y_test = load_moons_dataset()

# train under noise
noise_model = build_depolarizing_model(error_rate=0.01, n_qubits=2)
model = train_noisy_vqc(X_train, y_train, noise_model=noise_model)
acc = evaluate_classifier(model, X_test, y_test)
print(f"Noisy accuracy: {acc:.4f}")
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/error_mitigation_analysis.ipynb
```

## Results

### Accuracy vs Noise Level

| Depol. Error Rate | No Mitigation | ZNE | Meas. Mitigation | Combined |
|------------------|--------------|-----|-------------------|----------|
| 0.001 | 0.92 | 0.94 | 0.93 | 0.95 |
| 0.005 | 0.87 | 0.91 | 0.89 | 0.92 |
| 0.01 | 0.82 | 0.88 | 0.85 | 0.89 |
| 0.02 | 0.74 | 0.82 | 0.78 | 0.84 |
| 0.05 | 0.61 | 0.71 | 0.66 | 0.74 |

*Ideal (noiseless) accuracy: 0.96 on make_moons.*

### Key Findings

- Even small noise rates (1%) cause noticeable accuracy drops for VQC
- ZNE provides the most consistent improvement across all noise levels
- Measurement mitigation helps most when readout errors dominate
- Combining both techniques gives the best recovery, but with diminishing returns at high noise
- The mitigation overhead (extra circuit evaluations) is roughly 3-5x for ZNE

## References

1. Temme, K., Bravyi, S., & Gambetta, J. M. (2017). "Error Mitigation for Short-Depth Quantum Circuits." *Physical Review Letters*, 119(18), 180509.
2. Li, Y., & Benjamin, S. C. (2017). "Efficient Variational Quantum Simulator Incorporating Active Error Minimization." *Physical Review X*, 7(2), 021050.
3. Kandala, A., et al. (2019). "Error mitigation extends the computational reach of a noisy quantum processor." *Nature*, 567(7749), 491-495.

## License

MIT License — see [LICENSE](LICENSE) for details.
