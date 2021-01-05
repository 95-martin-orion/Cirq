# Copyright 2020 The Cirq Developers
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

A CircuitGate is a Gate object that wraps a FrozenCircuit. When applied as part
of a larger circuit, a CircuitGate will execute all component gates in order,
including any nested CircuitGates.
"""

import itertools
import sympy

from typing import (TYPE_CHECKING, AbstractSet, Any, Dict, List, Optional,
                    Sequence, Tuple, Union)

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
                 name: Optional[str] = None,
                 exp_modulus: int = 0) -> None:
        self._circuit = circuits.FrozenCircuit(contents,
                                               strategy=strategy,
                                               device=device)
        self._name = name
        self._exp_modulus = exp_modulus

    @property
    def circuit(self):
        return self._circuit

    @property
    def name(self):
        return self._name

    @property
    def exp_modulus(self):
        return self._exp_modulus

    def num_qubits(self):
        return protocols.num_qubits(self.circuit)

    def _qid_shape_(self):
        return protocols.qid_shape(self.circuit)

    def on(self, *qubits: 'cirq.Qid') -> 'CircuitOperation':
        return CircuitOperation(self, qubits)

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        new_gate = CircuitGate()
        new_gate._circuit = protocols.with_measurement_key_mapping(self.circuit,
                                                                   key_map)
        new_gate._name = self.name
        new_gate._exp_modulus = self.exp_modulus
        return new_gate

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.circuit == other.circuit

    def __hash__(self):
        return hash((self.circuit, self.name, self.exp_modulus))

    def __pow__(self, power):
        try:
            return CircuitGate(self.circuit**power, device=self.circuit.device)
        except:
            return NotImplemented

    def __repr__(self):
        return f'CircuitGate({repr(self.circuit)})'

    def __str__(self):
        header = "CircuitGate:"
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
        applied_circuit = self.circuit.unfreeze()
        if qubits is not None and qubits != self.ordered_qubits():
            qmap = {old: new for old, new in zip(self.ordered_qubits(), qubits)}
            applied_circuit = applied_circuit.transform_qubits(lambda q: qmap[q]
                                                              )

        return protocols.decompose(applied_circuit)

    def ordered_qubits(
            self,
            qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ):
        return ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.circuit.all_qubits())

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['circuit'])


class CircuitOperation(ops.GateOperation):
    """An operation containing a circuit and associated parameters.
    
    CircuitOperation is the fundamental "subroutine" object in Cirq: it
    contains a FrozenCircuit and accepts arguments (repetitions, parameters)
    to apply to that instance of the FrozenCircuit.
    """

    class LoopDescriptor:

        def __init__(self,
                     repetitions: int,
                     loop_param: Optional[sympy.Symbol] = None,
                     loop_values: Optional[List[float]] = None):
            self.repetitions = repetitions
            self.loop_param = loop_param
            self.loop_values = loop_values

    def __init__(self, gate: 'CircuitGate', qubits: Sequence['cirq.Qid']):
        super().__init__(gate, qubits)
        self._measurement_key_map: Dict[str, str] = {}
        self._param_values: Dict[sympy.Symbol, Any] = {}
        # Loops are stored innermost-first.
        self._loop_structure: List['CircuitOperation.LoopDescriptor'] = []
        self._inverted = False

    @property
    def circuit(self):
        return self.gate.circuit

    def __repr__(self):
        base_repr = super().__repr__()
        if self._measurement_key_map:
            base_repr += (
                f'.with_deferred_measurement_keys({self._measurement_key_map})')
        if self._param_values:
            base_repr += f'.with_deferred_params({self._param_values})'
        for loop in self._loop_structure:
            # TODO: fill out other loop fields
            base_repr += f'.with_loop({loop.repetitions})'
        return base_repr

    def base_operation(self):
        return CircuitOperation(self._gate, self._qubits)

    def copy(self):
        new_op = self.base_operation()
        new_op._measurement_key_map = self._measurement_key_map.copy()
        new_op._loop_structure = self._loop_structure.copy()
        new_op._param_values = self._param_values.copy()
        return new_op

    def with_gate(self, new_gate: 'cirq.Gate'):
        if not isinstance(new_gate, CircuitGate):
            return TypeError('CircuitOperations may only contain CircuitGates.')
        new_op = self.copy()
        new_op._gate = new_gate
        return new_op

    def with_deferred_measurement_keys(self, key_map: Dict[str, str]):
        new_op = self.copy()
        new_op._measurement_key_map = {
            k: (key_map[v] if v in key_map else v)
            for k, v in self._measurement_key_map.items()
        }
        new_op._measurement_key_map.update({
            key: val for key, val in key_map.items()
            if key not in self._measurement_key_map.values()
        })
        return new_op

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        return self.with_deferred_measurement_keys(key_map)

    def with_deferred_params(self, param_values: Dict[sympy.Symbol, Any]):
        new_op = self.copy()
        new_op._param_values = {
            key: (val if not protocols.is_parameterized(val) else
                  protocols.resolve_parameters(val, param_values))
            for key, val in self._param_values.items()
        }
        return new_op

    def with_loop(self,
                  repetitions: int,
                  loop_param: Optional[sympy.Symbol] = None,
                  loop_values: Optional[List[float]] = None):
        new_op = self.copy()
        new_op._loop_structure = [
            CircuitOperation.LoopDescriptor(repetitions, loop_param,
                                            loop_values)
        ] + self._loop_structure
        return new_op

    def invert(self):
        new_op = self.copy()
        new_op._inverted = not self._inverted
        return new_op

    def __pow__(self, power: int):
        if not isinstance(power, int):
            return NotImplemented
        if power < -1:
            return self.invert().with_loop(abs(power))
        if power == -1:
            return self.invert()
        if power == 1:
            return self.copy()
        return self.with_loop(power)

    # Methods below reference the decomposed form of the CircuitOperation.

    def _measurement_keys_(self):
        # TODO: this will misbehave with nested looping CircuitOperations.
        base_keys = protocols.measurement_keys(self.gate)
        mapped_keys = set(self._measurement_key_map[key] if key in
                          self._measurement_key_map else key
                          for key in base_keys)
        return self._unroll_measurement_keys(mapped_keys)

    def _unroll_measurement_keys(self, keys: AbstractSet[str]):
        if not self._loop_structure:
            return keys
        loop_indices = [[f'[{i}]'
                         for i in range(loop.repetitions)]
                        for loop in self._loop_structure]
        index_list = [
            ''.join(indices) for indices in itertools.product(*loop_indices)
        ]
        return set(f'{key}{indices}' for key in keys for indices in index_list)

    def _decompose_(self) -> 'cirq.OP_TREE':
        if self._inverted:
            all_ops = (self.gate.circuit**-1).all_operations()
        else:
            all_ops = self.gate.circuit.all_operations()

        if self._measurement_key_map:
            all_ops = [
                op if not protocols.is_measurement(op) else
                protocols.with_measurement_key_mapping(
                    op, self._measurement_key_map) for op in all_ops
            ]
        if self._param_values:
            all_ops = [
                op if not protocols.is_parameterized(op) else
                protocols.resolve_parameters(op, self._param_values)
                for op in all_ops
            ]
        all_ops = _unroll_loops(all_ops, self._loop_structure)
        return all_ops


def _unroll_loops(ops: Sequence['cirq.Operation'],
                  loops: Sequence['CircuitOperation.LoopDescriptor']
                 ) -> Sequence['cirq.Operation']:
    """Expands a series of loops over a list of operations.
    
    This method reassigns measurement keys and resolves the loop parameter for
    each loop in 'loops'.
    """
    if not loops:
        return ops

    unrolled_ops = []
    for i in range(loops[0].repetitions):
        loop_ops = [
            op if not protocols.is_measurement(op) else
            protocols.with_measurement_key_mapping(
                op, {
                    key: (f'{key}[{i}]' if '[' not in key else key.replace(
                        '[', f'[{i}][', 1))
                    for key in protocols.measurement_keys(op)
                })
            for op in ops
        ]
        if loops[0].loop_param:
            values = loops[0].loop_values or list(range(loops[0].repetitions))
            loop_ops = [
                op if not protocols.is_parameterized(op) else
                protocols.resolve_parameters(
                    op, {loops[0].loop_param: values[i]}) for op in loop_ops
            ]
        unrolled_ops.extend(loop_ops)
    return _unroll_loops(unrolled_ops, loops[1:])


# TODO: how do we handle tagged CircuitOperations?
# Needs to identify as both Tagged and Circuit op types
# (only if Circuit treats them differently in decompose...)
