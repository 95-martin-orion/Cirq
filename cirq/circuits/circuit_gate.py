# Copyright 2018 The Cirq Developers
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
"""A structure for encapsulating entire circuits in a gate.

CircuitGates consist of a FrozenCircuit and a repetition count. When applied
as part of a larger circuit, a CircuitGate will execute all component gates in
sequence, repeating as specified.
"""

from typing import TYPE_CHECKING, Any, Union

from cirq import circuits, devices, ops, protocols
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class CircuitGate(ops.Gate):
    """A single gate that encapsulates a Circuit."""

    def __init__(self,
                 *contents: 'cirq.OP_TREE',
                 strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST,
                 device: 'cirq.Device' = devices.UNCONSTRAINED_DEVICE,
                 repetitions: int = 1) -> None:
        self._circuit = circuits.FrozenCircuit(contents,
                                               strategy=strategy,
                                               device=device)
        self._repetitions = repetitions

    @property
    def circuit(self):
        return self._circuit

    @property
    def repetitions(self):
        return self._repetitions

    def repeat(self, repetitions: int) -> 'CircuitGate':
        if repetitions < 0 and not protocols.has_unitary(self.circuit):
            return None

        return CircuitGate(self.circuit,
                           device=self.circuit.device,
                           repetitions=repetitions * self.repetitions)

    def without_repetition(self) -> 'CircuitGate':
        return CircuitGate(self.circuit, repetitions=1)

    def num_qubits(self):
        return protocols.num_qubits(self.circuit)

    def _qid_shape_(self):
        return protocols.qid_shape(self.circuit)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return (self.circuit == other.circuit and
                self.repetitions == other.repetitions)

    def __pow__(self, power):
        if not isinstance(power, int):
            return NotImplemented
        return self.repeat(power)

    def __repr__(self):
        if self.repetitions == 1:
            return f'CircuitGate({repr(self.circuit)})'
        return f'CircuitGate({repr(self.circuit)}).repeat({self.repetitions})'

    def __str__(self):
        if self.repetitions == 1:
            header = 'CircuitGate:'
        elif self.repetitions > 0:
            header = f'CircuitGate (repeat {self.repetitions}x):'
        elif self.repetitions < 0:
            header = (
                f'CircuitGate (invert and repeat {abs(self.repetitions)}x):')

        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(header) - 4] + [len(line) for line in msg_lines])
        full_msg = '\n'.join([
            '[ {line:<{width}} ]'.format(line=line, width=msg_width)
            for line in msg_lines
        ])

        return header + '\n' + full_msg

    def _commutes_(self, other: Any,
                   atol: float) -> Union[None, NotImplementedType, bool]:
        return protocols.commutes(self.circuit, other, atol=atol)

    def _decompose_(self, qubits=None):
        if qubits is not None and qubits != self.ordered_qubits():
            qmap = {old: new for old, new in zip(self.ordered_qubits(), qubits)}
            applied_circuit = self.circuit.unfreeze().transform_qubits(lambda q:
                                                                       qmap[q])
        else:
            applied_circuit = self.circuit

        if self.repetitions < 0:
            applied_circuit = protocols.inverse(applied_circuit)
        return protocols.decompose(applied_circuit) * abs(self.repetitions)

    def ordered_qubits(
            self,
            qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ):
        return ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.circuit.all_qubits())

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['circuit', 'repetitions'])
