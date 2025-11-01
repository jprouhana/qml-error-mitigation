"""
Error mitigation implementations: Zero-Noise Extrapolation and
Measurement Error Mitigation.
"""

import numpy as np
from qiskit import QuantumCircuit


class ZeroNoiseExtrapolator:
    """
    Zero-Noise Extrapolation via gate folding.

    Gate folding replaces each gate G with G * G^dag * G to amplify noise
    by a factor of 3. For stretch_factor=5, we insert two G^dag*G pairs.
    """

    def amplify_noise(self, circuit, stretch_factor):
        """
        Amplify noise in a circuit by gate folding.
        stretch_factor should be an odd integer (1, 3, 5, ...).
        """
        if stretch_factor == 1:
            return circuit.copy()

        n_folds = (stretch_factor - 1) // 2
        folded = QuantumCircuit(circuit.num_qubits)

        for instruction in circuit.data:
            gate = instruction.operation
            qubits = instruction.qubits

            # original gate
            folded.append(gate, qubits)

            # insert gate-inverse-gate pairs
            for _ in range(n_folds):
                folded.append(gate.inverse(), qubits)
                folded.append(gate, qubits)

        return folded

    def extrapolate(self, expectation_values, stretch_factors, method='linear'):
        """
        Extrapolate to zero noise from measurements at different stretch factors.

        method: 'linear' for linear fit, 'quadratic' for quadratic fit
        """
        stretch_factors = np.array(stretch_factors, dtype=float)
        expectation_values = np.array(expectation_values, dtype=float)

        if method == 'linear':
            coeffs = np.polyfit(stretch_factors, expectation_values, 1)
            # extrapolate to stretch_factor = 0
            return np.polyval(coeffs, 0)
        elif method == 'quadratic':
            deg = min(2, len(stretch_factors) - 1)
            coeffs = np.polyfit(stretch_factors, expectation_values, deg)
            return np.polyval(coeffs, 0)
        else:
            raise ValueError(f"Unknown method: {method}")


class MeasurementMitigator:
    """
    Measurement error mitigation via calibration matrix inversion.

    Calibrates readout errors by preparing computational basis states and
    measuring them. The calibration matrix maps ideal -> noisy distributions,
    and we invert it to correct raw measurements.
    """

    def __init__(self):
        self.calibration_matrix = None

    def calibrate(self, n_qubits, backend, shots=8192):
        """
        Run calibration circuits to build the calibration matrix.
        Prepares each computational basis state and measures.
        """
        n_states = 2 ** n_qubits
        self.calibration_matrix = np.zeros((n_states, n_states))

        for state_idx in range(n_states):
            # prepare the target computational basis state
            bitstring = format(state_idx, f'0{n_qubits}b')
            qc = QuantumCircuit(n_qubits, n_qubits)

            for i, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    qc.x(i)

            qc.measure(range(n_qubits), range(n_qubits))

            # run and collect results
            job = backend.run(qc, shots=shots)
            counts = job.result().get_counts()

            for measured_str, count in counts.items():
                measured_idx = int(measured_str, 2)
                self.calibration_matrix[measured_idx, state_idx] = count / shots

    def apply(self, raw_counts, n_qubits):
        """
        Correct raw measurement counts using the calibration matrix.
        Uses least-squares pseudo-inverse to handle noisy calibration.
        """
        if self.calibration_matrix is None:
            raise RuntimeError("Must call calibrate() first")

        n_states = 2 ** n_qubits
        total_shots = sum(raw_counts.values())

        # convert counts to probability vector
        raw_probs = np.zeros(n_states)
        for bitstring, count in raw_counts.items():
            idx = int(bitstring, 2)
            raw_probs[idx] = count / total_shots

        # solve A * p_ideal = p_raw for p_ideal
        corrected_probs, _, _, _ = np.linalg.lstsq(
            self.calibration_matrix, raw_probs, rcond=None
        )

        # clip to valid probabilities
        corrected_probs = np.clip(corrected_probs, 0, 1)
        corrected_probs /= corrected_probs.sum()

        # convert back to counts
        corrected_counts = {}
        for idx in range(n_states):
            if corrected_probs[idx] > 1e-6:
                bitstring = format(idx, f'0{n_qubits}b')
                corrected_counts[bitstring] = int(round(corrected_probs[idx] * total_shots))

        return corrected_counts
