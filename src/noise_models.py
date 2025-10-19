from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import numpy as np


def build_depolarizing_model(error_rate, n_qubits):
    noise_model = NoiseModel()

    err1 = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(
        err1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't']
    )

    err2 = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(err2, ['cx'])

    return noise_model


def build_readout_error_model(error_rate, n_qubits):
    noise_model = NoiseModel()
    p = 1 - error_rate
    ro_err = ReadoutError([[p, error_rate], [error_rate, p]])
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])
    return noise_model


def build_combined_model(depol_rate, readout_rate, n_qubits):
    noise_model = NoiseModel()

    err1 = depolarizing_error(depol_rate, 1)
    noise_model.add_all_qubit_quantum_error(
        err1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't']
    )
    err2 = depolarizing_error(depol_rate, 2)
    noise_model.add_all_qubit_quantum_error(err2, ['cx'])

    p = 1 - readout_rate
    ro_err = ReadoutError([[p, readout_rate], [readout_rate, p]])
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])

    return noise_model


def get_noise_levels():
    return [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
