# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Sequence, TYPE_CHECKING

from cirq import devices, value, ops, protocols

from cirq.google import engine

if TYPE_CHECKING:
    import cirq


def _homogeneous_moment_is_measurements(moment: 'cirq.Moment') -> bool:
    """Whether the moment is nothing but measurement gates.

    If a moment is a mixture of measurement and non-measurement gates
    this will throw a ValueError.
    """
    cases = {protocols.is_measurement(gate) for gate in moment}
    if len(cases) == 2:
        raise ValueError("Moment must be homogeneous: all measurements "
                         "or all operations.")
    return True in cases


class DepolarizingNoiseModel(devices.NoiseModel):
    """Applies depolarizing noise to each qubit individually at the end of
    every moment.

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float):
        """A depolarizing noise model

        Args:
            depol_prob: Depolarizing probability.
        """
        value.validate_probability(depol_prob, 'depol prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if (_homogeneous_moment_is_measurements(moment) or
                self.is_virtual_moment(moment)):
            # coverage: ignore
            return moment

        return [
            moment,
            ops.Moment(
                self.qubit_noise_gate(q).with_tags(ops.VirtualTag())
                for q in system_qubits)
        ]


class ReadoutNoiseModel(devices.NoiseModel):
    """NoiseModel with probabilistic bit flips preceding measurement.

    This simulates readout error. Note that since noise is applied before the
    measurement moment, composing this model on top of another noise model will
    place the bit flips immediately before the measurement (regardless of the
    previously-added noise).

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, bitflip_prob: float):
        """A noise model with readout error.

        Args:
            bitflip_prob: Probability of a bit-flip during measurement.
        """
        value.validate_probability(bitflip_prob, 'bitflip prob')
        self.readout_noise_gate = ops.BitFlipChannel(bitflip_prob)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if self.is_virtual_moment(moment):
            return moment
        if _homogeneous_moment_is_measurements(moment):
            return [
                ops.Moment(
                    self.readout_noise_gate(q).with_tags(ops.VirtualTag())
                    for q in system_qubits), moment
            ]
        return moment


class DampedReadoutNoiseModel(devices.NoiseModel):
    """NoiseModel with T1 decay preceding measurement.

    This simulates asymmetric readout error. Note that since noise is applied
    before the measurement moment, composing this model on top of another noise
    model will place the T1 decay immediately before the measurement
    (regardless of the previously-added noise).

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, decay_prob: float):
        """A depolarizing noise model with damped readout error.

        Args:
            decay_prob: Probability of T1 decay during measurement.
        """
        value.validate_probability(decay_prob, 'decay_prob')
        self.readout_decay_gate = ops.AmplitudeDampingChannel(decay_prob)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if self.is_virtual_moment(moment):
            return moment
        if _homogeneous_moment_is_measurements(moment):
            return [
                ops.Moment(
                    self.readout_decay_gate(q).with_tags(ops.VirtualTag())
                    for q in system_qubits), moment
            ]
        return moment


class PerQubitDepolarizingNoiseModel(devices.NoiseModel):
    """DepolarizingNoiseModel which allows depolarization probabilities to be
    specified separately for each qubit.

    Similar to depol_prob in DepolarizingNoiseModel, depol_prob_map should map
    Qids in the device to their depolarization probability.
    """

    def __init__(
            self,
            depol_prob_map: Dict['cirq.Qid', float],
    ):
        """A depolarizing noise model with variable per-qubit noise.

        Args:
            depol_prob_map: Map of depolarizing probabilities for each qubit.
        """
        for qubit, depol_prob in depol_prob_map.items():
            value.validate_probability(depol_prob, f'depol prob of {qubit}')
        self.depol_prob_map = depol_prob_map

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        if (_homogeneous_moment_is_measurements(moment) or
                self.is_virtual_moment(moment)):
            return moment
        else:
            gated_qubits = [
                q for q in system_qubits if moment.operates_on_single_qubit(q)
            ]
            return [
                moment,
                ops.Moment(
                    ops.DepolarizingChannel(self.depol_prob_map[q])(q)
                    for q in gated_qubits)
            ]


class PerQubitDepolarizingWithDampedReadoutNoiseModel(devices.NoiseModel):
    """DepolarizingWithDampedReadoutNoiseModel which allows probabilities to be
    specified separately for each qubit.

    This simulates asymmetric readout error. The noise is structured
    so the T1 decay is applied, then the readout bitflip, then measurement.
    Note that T1 decay is only applied to measurement, not other gates.

    In moments without measurement, all qubits affected by an operation will
    have a depolarizing channel applied after the original operation. Qubits
    that remain idle will be unaffected by this model.

    As with the DepolarizingWithDampedReadoutNoiseModel, if a circuit contains
    measurements, they must be in moments that don't also contain gates.
    """

    def __init__(
            self,
            depol_prob_map: Dict['cirq.Qid', float] = None,
            bitflip_prob_map: Dict['cirq.Qid', float] = None,
            decay_prob_map: Dict['cirq.Qid', float] = None,
    ):
        """A depolarizing noise model with damped readout error.

        All error modes are specified on a per-qubit basis. To omit a given
        error mode from the noise model, leave its map blank when initializing
        this object.

        Args:
            depol_prob_map: Map of depolarizing probabilities for each qubit.
            bitflip_prob: Probability of a bit-flip during measurement.
            decay_prob: Probability of T1 decay during measurement.
                Bitflip noise is applied first, then amplitude decay.
        """
        for prob_map, desc in [(depol_prob_map, "depolarization prob"),
                               (bitflip_prob_map, "readout error prob"),
                               (decay_prob_map, "readout decay prob")]:
            if prob_map:
                for qubit, prob in prob_map.items():
                    value.validate_probability(prob, f'{desc} of {qubit}')
        self.depol_prob_map = depol_prob_map
        self.bitflip_prob_map = bitflip_prob_map
        self.decay_prob_map = decay_prob_map

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        moments = []
        if _homogeneous_moment_is_measurements(moment):
            if self.decay_prob_map:
                moments.append(
                    ops.Moment(
                        ops.AmplitudeDampingChannel(self.decay_prob_map[q])(q)
                        for q in system_qubits))
            if self.bitflip_prob_map:
                moments.append(
                    ops.Moment(
                        ops.BitFlipChannel(self.bitflip_prob_map[q])(q)
                        for q in system_qubits))
            moments.append(moment)
            return moments
        else:
            moments.append(moment)
            if self.depol_prob_map:
                gated_qubits = [
                    q for q in system_qubits
                    if moment.operates_on_single_qubit(q)
                ]
                moments.append(
                    ops.Moment(
                        ops.DepolarizingChannel(self.depol_prob_map[q])(q)
                        for q in gated_qubits))
            return moments


def simple_noise_from_calibration_metrics(calibration: engine.Calibration,
                                          depolNoise: bool = False,
                                          dampingNoise: bool = False,
                                          readoutDecayNoise: bool = False,
                                          readoutErrorNoise: bool = False
                                         ) -> devices.NoiseModel:
    """Creates a reasonable PerQubitDepolarizingWithDampedReadoutNoiseModel
    using the provided calibration data.

    Args:
        calibration: a Calibration object (cirq/google/engine/calibration.py).
            This object can be retrived from the engine by calling
            'get_latest_calibration()' or 'get_calibration()' using the ID of
            the target processor.
        depolNoise: Enables per-gate depolarization if True.
        dampingNoise: Enables per-gate amplitude damping if True.
            Currently unimplemented.
        readoutDecayNoise: Enables pre-readout amplitude damping if True.
        readoutErrorNoise: Enables pre-readout bitflip errors if True.

    Returns:
        A PerQubitDepolarizingWithDampedReadoutNoiseModel with error
            probabilities generated from the provided calibration data.
    """
    if not any([depolNoise, dampingNoise, readoutDecayNoise, readoutErrorNoise
               ]):
        raise ValueError('At least one error type must be specified.')
    assert calibration is not None
    depol_prob_map: Dict['cirq.Qid', float] = {}
    readout_decay_map: Dict['cirq.Qid', float] = {}
    readout_error_map: Dict['cirq.Qid', float] = {}

    if depolNoise:
        depol_prob_map = {
            qubit[0]: depol_prob[0] for qubit, depol_prob in
            calibration['single_qubit_rb_total_error'].items()
        }
    if dampingNoise:
        # TODO: implement per-gate amplitude damping noise.
        raise NotImplementedError('Gate damping is not yet supported.')

    if readoutDecayNoise:
        # Copied from Sycamore readout duration in known_devices.py
        # TODO: replace with polling from DeviceSpecification.
        readout_micros = 1
        readout_decay_map = {
            qubit[0]: exp(1 - readout_micros / t1[0])
            for qubit, t1 in calibration['single_qubit_idle_t1_micros'].items()
        }
    if readoutErrorNoise:
        # This assumes that p(<1|0>) is negligible for readout timescales.
        readout_error_map = {
            qubit[0]: p0[0] for qubit, p0 in
            calibration['single_qubit_readout_p0_error'].items()
        }
    return PerQubitDepolarizingWithDampedReadoutNoiseModel(
        depol_prob_map=depol_prob_map,
        decay_prob_map=readout_decay_map,
        bitflip_prob_map=readout_error_map)
