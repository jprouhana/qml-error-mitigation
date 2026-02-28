# qml-error-mitigation

Testing zero-noise extrapolation and measurement error mitigation on a noisy VQC. Lets see how much accuracy we can recover at different error rates.

## setup

```
pip install -r requirements.txt
```

## usage

```python
from src.noise_models import build_depolarizing_model
from src.noisy_classifier import train_noisy_vqc, evaluate_classifier
from src.data_utils import load_moons_dataset

X_train, X_test, y_train, y_test = load_moons_dataset()
noise = build_depolarizing_model(0.01, n_qubits=2)
model = train_noisy_vqc(X_train, y_train, noise_model=noise)
```

full analysis in `notebooks/error_mitigation_analysis.ipynb`.
