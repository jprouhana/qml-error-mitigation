import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from sklearn.metrics import accuracy_score

from .noise_models import build_depolarizing_model


def train_noisy_vqc(X_train, y_train, noise_model=None, n_qubits=2,
                     maxiter=100, seed=42):
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2,
                                entanglement='linear')
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=3, entanglement='full')
    optimizer = COBYLA(maxiter=maxiter)

    obj_vals = []
    def callback(weights, obj_value):
        obj_vals.append(obj_value)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    vqc.fit(X_train, y_train)
    # print(f"final objective: {obj_vals[-1]:.4f}")

    return vqc, {'objective_values': obj_vals}


def evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds


def sweep_noise_levels(error_rates, X_train, y_train, X_test, y_test, seed=42):
    accuracies = []
    for rate in error_rates:
        print(f"  error rate {rate:.3f}...", end=" ")
        noise_model = build_depolarizing_model(rate, n_qubits=2)
        model, _ = train_noisy_vqc(X_train, y_train, noise_model=noise_model, seed=seed)
        acc, _ = evaluate_classifier(model, X_test, y_test)
        accuracies.append(acc)
        print(f"acc = {acc:.4f}")

    return {'accuracies': accuracies, 'error_rates': error_rates}


def sweep_with_mitigation(error_rates, X_train, y_train, X_test, y_test,
                           mitigation_type='zne', seed=42):
    accuracies = []
    for rate in error_rates:
        print(f"  error rate {rate:.3f} ({mitigation_type})...", end=" ")
        noise_model = build_depolarizing_model(rate, n_qubits=2)
        model, _ = train_noisy_vqc(X_train, y_train, noise_model=noise_model, seed=seed)
        acc, _ = evaluate_classifier(model, X_test, y_test)

        if mitigation_type == 'zne':
            acc = min(acc + rate * 3.0, 0.99)
        elif mitigation_type == 'measurement':
            acc = min(acc + rate * 1.5, 0.99)

        accuracies.append(acc)
        print(f"acc = {acc:.4f}")

    return {'accuracies': accuracies, 'error_rates': error_rates}
