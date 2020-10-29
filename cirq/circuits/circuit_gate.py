
from typing import TYPE_CHECKING, Any, Union

from cirq import circuits, devices, ops, protocols
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

class CircuitGate(ops.Gate):

    def __init__(self,
                 *contents: 'cirq.OP_TREE',
                 strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST,
                 device: 'cirq.Device' = devices.UNCONSTRAINED_DEVICE) -> None:
        self._circuit = circuits.FrozenCircuit(contents,
                                               strategy=strategy,
                                               device=device)

    @property
    def circuit(self):
        return self._circuit

    def num_qubits(self):
        return protocols.num_qubits(self._circuit)

    def _qid_shape_(self):
        return protocols.qid_shape(self._circuit)

    def __eq__(self, other):
        return self._circuit == other._circuit

    def __pow__(self, power):
        # TODO: special pow handling for circuit
        try:
            return CircuitGate(self._circuit**power,
                               strategy=InsertStrategy.EARLIEST,
                               device=self._circuit.device)
        except:
            return NotImplemented

    # TODO: reconsider string ops
    def __repr__(self):
        return f'CircuitGate({repr(self._circuit)})'

    def __str__(self):
        return f'CircuitGate\n{self._circuit})'

    def _commutes_(self, other: Any,
                   atol: float) -> Union[None, NotImplementedType, bool]:
        return protocols.commutes(self._circuit, other, atol=atol)

    def _decompose_(self, qubits=None):
        if qubits is None or qubits == self.ordered_qubits():
            return protocols.decompose(self._circuit)
        
        # TODO: broken (all_qubits is an unordered frozenset)
        # Existence of qid_shape implies an ordering exists
        qmap = {
            old: new for old, new in 
            zip(self.ordered_qubits(), qubits)
        }
        applied_circuit = self._circuit.unfreeze().transform_qubits(
            lambda q: qmap[q])
        return protocols.decompose(applied_circuit)

    def ordered_qubits(
            self,
            qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ):
        return ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self._circuit.all_qubits())

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['circuit'])
