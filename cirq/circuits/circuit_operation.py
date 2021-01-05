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
"""An operation representing a circuit and attached parameters.

A CircuitOperation is a GateOperation that contains a CircuitGate. It also
captures "arguments" to the contained circuit, such as number of repetitions
and parameter values.
"""

from typing import TYPE_CHECKING, Sequence

from cirq import circuits, ops

if TYPE_CHECKING:
    import cirq


class CircuitOperation(ops.GateOperation):
    
    def __init__(self,
                 gate: circuits.CircuitGate,
                 qubits: Sequence['cirq.Qid']):
        super().__init__(gate, qubits)
