"""
Noise model construction for simulating errors on quantum circuits.
"""

from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import numpy as np


def build_depolarizing_model(error_rate, n_qubits):
    """Create a depolarizing noise model for all gates."""
    noise_model = NoiseModel()
    error_1q = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(
        error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't']
    )
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model


def build_readout_error_model(error_rate, n_qubits):
    """Create a noise model with only measurement readout errors."""
    noise_model = NoiseModel()
    p_correct = 1 - error_rate
    readout_err = ReadoutError([[p_correct, error_rate], [error_rate, p_correct]])
    for qubit in range(n_qubits):
        noise_model.add_readout_error(readout_err, [qubit])
    return noise_model


def build_combined_model(depol_rate, readout_rate, n_qubits):
    """Build a noise model with both depolarizing and readout errors."""
    noise_model = NoiseModel()
    error_1q = depolarizing_error(depol_rate, 1)
    noise_model.add_all_qubit_quantum_error(
        error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't']
    )
    error_2q = depolarizing_error(depol_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    p_correct = 1 - readout_rate
    readout_err = ReadoutError([[p_correct, readout_rate], [readout_rate, p_correct]])
    for qubit in range(n_qubits):
        noise_model.add_readout_error(readout_err, [qubit])
    return noise_model


def get_noise_levels():
    """Standard error rates for noise sweeps."""
    return [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
