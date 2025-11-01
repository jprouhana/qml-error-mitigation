import numpy as np
from qiskit import QuantumCircuit


class ZeroNoiseExtrapolator:

    def amplify_noise(self, circuit, stretch_factor):
        if stretch_factor == 1:
            return circuit.copy()

        n_folds = (stretch_factor - 1) // 2
        folded = QuantumCircuit(circuit.num_qubits)

        for instruction in circuit.data:
            gate = instruction.operation
            qubits = instruction.qubits

            folded.append(gate, qubits)

            for _ in range(n_folds):
                folded.append(gate.inverse(), qubits)
                folded.append(gate, qubits)

        return folded

    def extrapolate(self, expectation_values, stretch_factors, method='linear'):
        stretch_factors = np.array(stretch_factors, dtype=float)
        expectation_values = np.array(expectation_values, dtype=float)

        # print(f"extrapolating with {method}, factors={stretch_factors}")

        if method == 'linear':
            coeffs = np.polyfit(stretch_factors, expectation_values, 1)
            return np.polyval(coeffs, 0)
        elif method == 'quadratic':
            deg = min(2, len(stretch_factors) - 1)
            coeffs = np.polyfit(stretch_factors, expectation_values, deg)
            return np.polyval(coeffs, 0)
        else:
            raise ValueError(f"Unknown method: {method}")


class MeasurementMitigator:

    def __init__(self):
        self.calibration_matrix = None

    def calibrate(self, n_qubits, backend, shots=8192):
        n_states = 2 ** n_qubits
        self.calibration_matrix = np.zeros((n_states, n_states))

        for state_idx in range(n_states):
            bitstring = format(state_idx, f'0{n_qubits}b')
            qc = QuantumCircuit(n_qubits, n_qubits)

            for i, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    qc.x(i)

            qc.measure(range(n_qubits), range(n_qubits))

            # print(f"calibrating state |{bitstring}>")
            job = backend.run(qc, shots=shots)
            counts = job.result().get_counts()

            for measured_str, count in counts.items():
                measured_idx = int(measured_str, 2)
                self.calibration_matrix[measured_idx, state_idx] = count / shots

    def apply(self, raw_counts, n_qubits):
        if self.calibration_matrix is None:
            raise RuntimeError("Must call calibrate() first")

        n_states = 2 ** n_qubits
        total_shots = sum(raw_counts.values())

        raw_probs = np.zeros(n_states)
        for bitstring, count in raw_counts.items():
            idx = int(bitstring, 2)
            raw_probs[idx] = count / total_shots

        corrected_probs, _, _, _ = np.linalg.lstsq(
            self.calibration_matrix, raw_probs, rcond=None
        )

        corrected_probs = np.clip(corrected_probs, 0, 1)
        corrected_probs /= corrected_probs.sum()

        corrected_counts = {}
        for idx in range(n_states):
            if corrected_probs[idx] > 1e-6:
                bitstring = format(idx, f'0{n_qubits}b')
                corrected_counts[bitstring] = int(round(corrected_probs[idx] * total_shots))

        return corrected_counts
