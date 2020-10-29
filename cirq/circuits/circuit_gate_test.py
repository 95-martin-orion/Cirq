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

from typing import AbstractSet

import pytest
import numpy as np
import sympy

import cirq


def test_circuit_gate():
    a, b, c = cirq.LineQubit.range(3)

    g = cirq.CircuitGate(cirq.CZ(a, b))
    assert cirq.num_qubits(g) == 2

    _ = g.on(a, c)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g.on(a, c, b)

    _ = g(a, c)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g(a)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g(c, b, a)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = g(a, b.with_dimension(3))

    assert g.controlled(0) is g


def test_circuit_op():
    a, b, c = cirq.LineQubit.range(3)
    g = cirq.CircuitGate(cirq.X(a))
    op = g(a)
    assert op.controlled_by() is op
    controlled_op = op.controlled_by(b, c)
    assert controlled_op.sub_operation == op
    assert controlled_op.controls == (b, c)


def test_circuit_op_validate():
    cg = cirq.CircuitGate(cirq.X(cirq.NamedQubit('placeholder')))
    op = cg.on(cirq.LineQid(0, 2))
    cg2 = cirq.CircuitGate(cirq.CNOT(*cirq.LineQubit.range(2)))
    op2 = cg2.on(*cirq.LineQid.range(2, dimension=2))
    op.validate_args([cirq.LineQid(1, 2)])  # Valid
    op2.validate_args(cirq.LineQid.range(1, 3, dimension=2))  # Valid
    with pytest.raises(ValueError, match='Wrong shape'):
        op.validate_args([cirq.LineQid(1, 9)])
    with pytest.raises(ValueError, match='Wrong number'):
        op.validate_args([cirq.LineQid(1, 2), cirq.LineQid(2, 2)])
    with pytest.raises(ValueError, match='Duplicate'):
        op2.validate_args([cirq.LineQid(1, 2), cirq.LineQid(1, 2)])


def test_default_validation_and_inverse():
    a, b = cirq.LineQubit.range(2)
    cg = cirq.CircuitGate(cirq.Z(a), cirq.S(b), cirq.X(a))

    with pytest.raises(ValueError, match='number of qubits'):
        cg.on(a)

    t = cg.on(a, b)
    i = t**-1
    assert i**-1 == t
    assert t**-1 == i
    print(cirq.Circuit(t).moments)
    print(cirq.Circuit(i).moments)
    assert cirq.Circuit(cirq.decompose(i)) == cirq.Circuit(
        cirq.X(a),
        cirq.S(b)**-1, cirq.Z(a))
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(i),
                                                    cirq.unitary(t).conj().T,
                                                    atol=1e-8)

    cirq.testing.assert_implements_consistent_protocols(
        i, local_vals={'CircuitGate': cirq.CircuitGate})


def test_default_inverse():
    qubits = cirq.LineQubit.range(3)
    cg = cirq.CircuitGate(cirq.X(q) for q in qubits)

    assert cirq.inverse(cg, None) is not None
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(cg))
    cirq.testing.assert_has_consistent_qid_shape(
        cirq.inverse(cg.on(*cirq.LineQubit.range(3))))


def test_repetition_and_inverse():
    a, b = cirq.LineQubit.range(2)
    cg = cirq.CircuitGate(cirq.Z(a), cirq.S(b), cirq.X(a))

    t = (cg**3).on(a, b)
    i = t**-1
    assert cirq.Circuit(cirq.decompose(t)) == cirq.Circuit(
        [cirq.Z(a), cirq.S(b), cirq.X(a)] * 3)
    assert cirq.Circuit(cirq.decompose(i)) == cirq.Circuit(
        [cirq.X(a), cirq.S(b)**-1, cirq.Z(a)] * 3)

    five_t = t**5
    assert cirq.Circuit(cirq.decompose(five_t)) == cirq.Circuit(
        [cirq.Z(a), cirq.S(b), cirq.X(a)] * 3 * 5)


def test_no_inverse_if_not_unitary():
    cg = cirq.CircuitGate(cirq.amplitude_damp(0.5).on(cirq.LineQubit(0)))
    assert cirq.inverse(cg, None) is None


def test_default_qudit_inverse():
    q = cirq.LineQid.for_qid_shape((1, 2, 3))
    cg = cirq.CircuitGate(
        cirq.IdentityGate(qid_shape=(1,)).on(q[0]),
        (cirq.X**0.1).on(q[1]),
        cirq.IdentityGate(qid_shape=(3,)).on(q[2]),
    )

    assert cirq.qid_shape(cg.on(*q)) == (1, 2, 3)
    assert cirq.qid_shape(cirq.inverse(cg, None)) == (1, 2, 3)
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(cg))


def test_circuit_gate_shape():
    shape_gate = cirq.CircuitGate(
        cirq.IdentityGate(qid_shape=(q.dimension,)).on(q)
        for q in cirq.LineQid.for_qid_shape((1, 2, 3, 4)))
    assert cirq.qid_shape(shape_gate) == (1, 2, 3, 4)
    assert cirq.num_qubits(shape_gate) == 4
    assert shape_gate.num_qubits() == 4

    qubit_gate = cirq.CircuitGate(cirq.I(q) for q in cirq.LineQubit.range(3))
    assert cirq.qid_shape(qubit_gate) == (2, 2, 2)
    assert cirq.num_qubits(qubit_gate) == 3
    assert qubit_gate.num_qubits() == 3


def test_circuit_gate_json_dict():
    cg = cirq.CircuitGate(cirq.X(cirq.LineQubit(0)))
    assert cg._json_dict_() == {
        'cirq_type': 'CircuitGate',
        'circuit': cg.circuit,
        'repetitions': 1
    }

    cg = cg**-3
    assert cg._json_dict_() == {
        'cirq_type': 'CircuitGate',
        'circuit': cg.circuit,
        'repetitions': -3
    }


def test_string_formats():
    x, y, z = cirq.LineQubit.range(3)

    cg = cirq.CircuitGate(cirq.X(x), cirq.H(y), cirq.CX(y, z),
                          cirq.measure(x, y, z, key='m'))

    assert str(cg) == """\
CircuitGate:
[ 0: ───X───────M('m')─── ]
[               │         ]
[ 1: ───H───@───M──────── ]
[           │   │         ]
[ 2: ───────X───M──────── ]"""

    assert str(cg**5) == """\
CircuitGate (repeat 5x):
[ 0: ───X───────M('m')─── ]
[               │         ]
[ 1: ───H───@───M──────── ]
[           │   │         ]
[ 2: ───────X───M──────── ]"""

    assert str(cg**-3) == """\
CircuitGate (invert and repeat 3x):
[ 0: ───X───────M('m')───         ]
[               │                 ]
[ 1: ───H───@───M────────         ]
[           │   │                 ]
[ 2: ───────X───M────────         ]"""

    cs = cirq.Circuit(cirq.Z(z), (cg**-3).on(x, y, z))
    assert str(cs) == """\
          CircuitGate (invert and repeat 3x):
          [ 0: ───X───────M('m')───         ]
0: ───────[               │                 ]───
          [ 1: ───H───@───M────────         ]
          [           │   │                 ]
          [ 2: ───────X───M────────         ]
          │
1: ───────#2────────────────────────────────────
          │
2: ───Z───#3────────────────────────────────────"""


# TODO: test CircuitGates in Circuits
