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
"""Support for serializing and deserializing cirq.google.api.v2 protos."""

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

from cirq import circuits, ops
from cirq._compat import deprecated, deprecated_parameter
from cirq.google import op_deserializer, op_serializer, arg_func_langs
from cirq.google.api import v2

if TYPE_CHECKING:
    import cirq

# Fake gate ID used to trigger deserialization of generic operations.
FALLBACK_TO_GENERIC = 'fallback_to_generic'


class SerializableGateSet:
    """A class for serializing and deserializing programs and operations.

    This class is for cirq.google.api.v2. protos.
    """

    def __init__(
        self,
        gate_set_name: str,
        serializers: Iterable[op_serializer.OpSerializer],
        deserializers: Iterable[op_deserializer.OpDeserializer],
    ):
        """Construct the gate set.

        Args:
            gate_set_name: The name used to identify the gate set.
            serializers: The OpSerializers to use for serialization.
                Multiple serializers for a given gate type are allowed and
                will be checked for a given type in the order specified here.
                This allows for a given gate type to be serialized into
                different serialized form depending on the parameters of the
                gate.
            deserializers: The OpDeserializers to convert serialized
                forms of gates or circuits into Operations.
        """
        self.gate_set_name = gate_set_name
        self.serializers: Dict[Type, List[op_serializer.OpSerializer]] = {}
        for s in serializers:
            self.serializers.setdefault(s.internal_type, []).append(s)
        self.deserializers = {d.serialized_id: d for d in deserializers}

    def with_added_types(
        self,
        *,
        gate_set_name: Optional[str] = None,
        serializers: Iterable[op_serializer.OpSerializer] = (),
        deserializers: Iterable[op_deserializer.OpDeserializer] = (),
    ) -> 'SerializableGateSet':
        """Creates a new gateset with additional (de)serializers.

        Args:
            gate_set_name: Optional new name of the gateset. If not given, use
                the same name as this gateset.
            serializers: Serializers to add to those in this gateset.
            deserializers: Deserializers to add to those in this gateset.
        """
        # Iterate over all serializers in this gateset.
        curr_serializers = (
            serializer for serializers in self.serializers.values() for serializer in serializers
        )
        return SerializableGateSet(
            gate_set_name or self.gate_set_name,
            serializers=[*curr_serializers, *serializers],
            deserializers=[*self.deserializers.values(), *deserializers],
        )

    @deprecated(deadline='v0.12.0', fix='Use with_added_types instead.')
    def with_added_gates(
        self,
        *,
        gate_set_name: Optional[str] = None,
        serializers: Iterable[op_serializer.OpSerializer] = (),
        deserializers: Iterable[op_deserializer.OpDeserializer] = (),
    ) -> 'SerializableGateSet':
        return self.with_added_types(
            gate_set_name=gate_set_name,
            serializers=serializers,
            deserializers=deserializers,
        )

    def supported_internal_types(self) -> Tuple:
        return tuple(self.serializers.keys())

    @deprecated(deadline='v0.12.0', fix='Use supported_internal_types instead.')
    def supported_gate_types(self) -> Tuple:
        return self.supported_internal_types()

    def is_supported(self, op_tree: 'cirq.OP_TREE') -> bool:
        """Whether the given object contains only supported operations."""
        return all(self.is_supported_operation(op) for op in ops.flatten_to_ops(op_tree))

    def is_supported_operation(self, op: 'cirq.Operation') -> bool:
        """Whether or not the given gate can be serialized by this gate set."""
        subcircuit = getattr(op.untagged, 'circuit', None)
        if subcircuit is not None:
            return self.is_supported(subcircuit)
        return any(
            serializer.can_serialize_operation(op)
            for gate_type in type(op.gate).mro()
            for serializer in self.serializers.get(gate_type, [])
        )

    @deprecated_parameter(
        deadline='v0.12.0',
        fix='Use use_constants instead.',
        parameter_desc='keyword use_constants_table_for_tokens',
        match=lambda args, kwargs: 'use_constants_table_for_tokens' in kwargs,
        rewrite=lambda args, kwargs: (
            args,
            {
                ('use_constants' if k == 'use_constants_table_for_tokens' else k): v
                for k, v in kwargs.items()
            },
        ),
    )
    def serialize(
        self,
        program: 'cirq.Circuit',
        msg: Optional[v2.program_pb2.Program] = None,
        *,
        arg_function_language: Optional[str] = None,
        use_constants: bool = True,
    ) -> v2.program_pb2.Program:
        """Serialize a Circuit to cirq.google.api.v2.Program proto.

        Args:
            program: The Circuit to serialize.
        """
        if msg is None:
            msg = v2.program_pb2.Program()
        msg.language.gate_set = self.gate_set_name
        if isinstance(program, circuits.Circuit):
            constants: Optional[List[v2.program_pb2.Constant]] = [] if use_constants else None
            raw_constants: Optional[List[Any]] = [] if use_constants else None
            self._serialize_circuit(
                program,
                msg.circuit,
                arg_function_language=arg_function_language,
                constants=constants,
                raw_constants=raw_constants,
            )
            if constants is not None:
                msg.constants.extend(constants)
            if arg_function_language is None:
                arg_function_language = arg_func_langs._infer_function_language_from_circuit(
                    msg.circuit
                )
        else:
            raise NotImplementedError(f'Unrecognized program type: {type(program)}')
        msg.language.arg_function_language = arg_function_language
        return msg

    def serialize_op(
        self,
        op: 'cirq.Operation',
        msg: Optional[v2.program_pb2.GenericOperation] = None,
        *,
        arg_function_language: Optional[str] = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[List[Any]] = None,
    ) -> v2.program_pb2.Operation:
        """Serialize an Operation to cirq.google.api.v2.Operation proto.

        Args:
            op: The operation to serialize.

        Returns:
            A dictionary corresponds to the cirq.google.api.v2.Operation proto.
        """
        circuit = getattr(op.untagged, 'circuit', None)
        if circuit is not None:
            if constants is None or raw_constants is None:
                raise ValueError(
                    'CircuitOp serialization requires a constants list and a corresponding '
                    'list of pre-serialization values (raw_constants).'
                )
            if circuits.FrozenCircuit in self.serializers:
                serializer = self.serializers[circuits.FrozenCircuit][0]
                if circuit not in raw_constants:
                    subcircuit_msg = v2.program_pb2.Circuit()
                    self._serialize_circuit(
                        circuit,
                        subcircuit_msg,
                        arg_function_language=arg_function_language,
                        constants=constants,
                        raw_constants=raw_constants,
                    )
                    constants.append(v2.program_pb2.Constant(circuit_value=subcircuit_msg))
                    raw_constants.append(circuit)
                proto_msg = serializer.to_proto(
                    op,
                    msg.circuit_operation if msg else None,
                    arg_function_language=arg_function_language,
                    constants=constants,
                    raw_constants=raw_constants,
                )
                if proto_msg is not None:
                    return proto_msg
            raise ValueError(f'Cannot serialize CircuitOperation {op!r}')

        gate_type = type(op.gate)
        for gate_type_mro in gate_type.mro():
            # Check all super classes in method resolution order.
            if gate_type_mro in self.serializers:
                # Check each serializer in turn, if serializer proto returns
                # None, then skip.
                for serializer in self.serializers[gate_type_mro]:
                    proto_msg = serializer.to_proto(
                        op,
                        msg.operation if msg else None,
                        arg_function_language=arg_function_language,
                        constants=constants,
                        raw_constants=raw_constants,
                    )
                    if proto_msg is not None:
                        return proto_msg
        raise ValueError(f'Cannot serialize op {op!r} of type {gate_type}')

    def deserialize(
        self, proto: v2.program_pb2.Program, device: Optional['cirq.Device'] = None
    ) -> 'cirq.Circuit':
        """Deserialize a Circuit from a cirq.google.api.v2.Program.

        Args:
            proto: A dictionary representing a cirq.google.api.v2.Program proto.
            device: If the proto is for a schedule, a device is required
                Otherwise optional.

        Returns:
            The deserialized Circuit, with a device if device was
            not None.
        """
        if not proto.HasField('language') or not proto.language.gate_set:
            raise ValueError('Missing gate set specification.')
        if proto.language.gate_set != self.gate_set_name:
            raise ValueError(
                'Gate set in proto was {} but expected {}'.format(
                    proto.language.gate_set, self.gate_set_name
                )
            )
        which = proto.WhichOneof('program')
        if which == 'circuit':
            raw_constants: List[Any] = []
            for constant in proto.constants:
                which_const = constant.WhichOneof('const_value')
                if which_const == 'string_value':
                    raw_constants.append(constant.string_value)
                elif which_const == 'circuit_value':
                    circuit = self._deserialize_circuit(
                        constant.circuit_value,
                        arg_function_language=proto.language.arg_function_language,
                        constants=proto.constants,
                        raw_constants=raw_constants,
                    )
                    raw_constants.append(circuit.freeze())
            circuit = self._deserialize_circuit(
                proto.circuit,
                arg_function_language=proto.language.arg_function_language,
                constants=proto.constants,
                raw_constants=raw_constants,
            )
            return circuit if device is None else circuit.with_device(device)
        if which == 'schedule':
            if device is None:
                raise ValueError('Deserializing schedule requires a device but None was given.')
            return self._deserialize_schedule(
                proto.schedule, device, arg_function_language=proto.language.arg_function_language
            )

        raise NotImplementedError('Program proto does not contain a circuit.')

    def deserialize_op(
        self,
        operation_proto: Union[
            v2.program_pb2.Operation,
            v2.program_pb2.CircuitOperation,
            v2.program_pb2.GenericOperation,
        ],
        **kwargs,
    ) -> 'cirq.Operation':
        """Disambiguation for operation deserialization."""
        if isinstance(operation_proto, v2.program_pb2.GenericOperation):
            which = operation_proto.WhichOneof('op')
            if which == 'operation':
                return self.deserialize_gate_op(operation_proto.operation, **kwargs)
            if which == 'circuit_operation':
                return self.deserialize_circuit_op(operation_proto.circuit_operation, **kwargs)

        if isinstance(operation_proto, v2.program_pb2.Operation):
            return self.deserialize_gate_op(operation_proto, **kwargs)

        if isinstance(operation_proto, v2.program_pb2.CircuitOperation):
            return self.deserialize_circuit_op(operation_proto, **kwargs)

        raise NotImplementedError('Program proto does not contain a circuit.')

    def deserialize_gate_op(
        self,
        operation_proto: v2.program_pb2.Operation,
        *,
        arg_function_language: str = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[List[Any]] = None,
    ) -> 'cirq.Operation':
        """Deserialize an Operation from a cirq.google.api.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.Operation proto.

        Returns:
            The deserialized Operation.
        """
        if not operation_proto.gate.id:
            raise ValueError('Operation proto does not have a gate.')

        gate_id = operation_proto.gate.id
        deserializer = self.deserializers.get(gate_id, None)
        if deserializer is None:
            raise ValueError(
                f'Unsupported serialized gate with id "{gate_id}".'
                f'\n\noperation_proto:\n{operation_proto}'
            )

        return deserializer.from_proto(
            operation_proto,
            arg_function_language=arg_function_language,
            constants=constants,
            raw_constants=raw_constants,
        )

    def deserialize_circuit_op(
        self,
        operation_proto: v2.program_pb2.CircuitOperation,
        *,
        arg_function_language: str = '',
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[List[Any]] = None,
    ) -> 'cirq.CircuitOperation':
        """Deserialize a CircuitOperation from a
            cirq.google.api.v2.CircuitOperation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.CircuitOperation proto.

        Returns:
            The deserialized CircuitOperation.
        """
        deserializer = self.deserializers.get('circuit', None)
        if deserializer is None:
            raise ValueError(
                f'Unsupported serialized CircuitOperation.\n\noperation_proto:\n{operation_proto}'
            )

        if not isinstance(deserializer, op_deserializer.CircuitOpDeserializer):
            raise ValueError(f'Deserializer {deserializer} cannot deserialize CircuitOperations.')

        return deserializer.from_proto(
            operation_proto,
            arg_function_language=arg_function_language,
            constants=constants,
            raw_constants=raw_constants,
        )

    def _serialize_circuit(
        self,
        circuit: 'cirq.AbstractCircuit',
        msg: v2.program_pb2.Circuit,
        *,
        arg_function_language: Optional[str],
        constants: Optional[List[v2.program_pb2.Constant]] = None,
        raw_constants: Optional[List[Any]] = None,
    ) -> None:
        msg.scheduling_strategy = v2.program_pb2.Circuit.MOMENT_BY_MOMENT
        for moment in circuit:
            moment_proto = msg.moments.add()
            for op in moment:
                gen_op_pb = moment_proto.generic_operations.add()
                self.serialize_op(
                    op,
                    gen_op_pb,
                    arg_function_language=arg_function_language,
                    constants=constants,
                    raw_constants=raw_constants,
                )
                op_pb = moment_proto.operations.add()
                which = gen_op_pb.WhichOneof('op')
                if which == 'operation':
                    op_pb.CopyFrom(gen_op_pb.operation)
                elif which == 'circuit_operation':
                    # Enables fallback to generic ops in deserialization.
                    op_pb.gate.id = FALLBACK_TO_GENERIC

    def _deserialize_circuit(
        self,
        circuit_proto: v2.program_pb2.Circuit,
        *,
        arg_function_language: str,
        constants: List[v2.program_pb2.Constant],
        raw_constants: List[Any],
    ) -> 'cirq.Circuit':
        moments = []
        for i, moment_proto in enumerate(circuit_proto.moments):
            moment_ops = []
            try_generic = False
            for op in moment_proto.operations:
                try:
                    moment_ops.append(
                        self.deserialize_op(
                            op,
                            arg_function_language=arg_function_language,
                            constants=constants,
                            raw_constants=raw_constants,
                        )
                    )
                except ValueError as ex:
                    if op.gate.id == FALLBACK_TO_GENERIC:
                        try_generic = True
                    else:
                        raise ValueError(
                            f'Failed to deserialize circuit. '
                            f'There was a problem in moment {i} '
                            f'handling an operation with the '
                            f'following proto:\n{op}'
                        ) from ex
            if try_generic:
                moment_ops = []
                for gen_op in moment_proto.generic_operations:
                    try:
                        moment_ops.append(
                            self.deserialize_op(
                                gen_op,
                                arg_function_language=arg_function_language,
                                constants=constants,
                                raw_constants=raw_constants,
                            )
                        )
                    except ValueError as ex:
                        raise ValueError(
                            f'Failed to deserialize circuit. '
                            f'There was a problem in moment {i} '
                            f'handling a generic operation with the '
                            f'following proto:\n{gen_op}'
                        ) from ex
            moments.append(ops.Moment(moment_ops))
        return circuits.Circuit(moments)

    def _deserialize_schedule(
        self,
        schedule_proto: v2.program_pb2.Schedule,
        device: 'cirq.Device',
        *,
        arg_function_language: str,
    ) -> 'cirq.Circuit':
        result = []
        for scheduled_op_proto in schedule_proto.scheduled_operations:
            if not scheduled_op_proto.HasField('operation'):
                raise ValueError('Scheduled op missing an operation {}'.format(scheduled_op_proto))
            result.append(
                self.deserialize_op(
                    scheduled_op_proto.operation, arg_function_language=arg_function_language
                )
            )
        return circuits.Circuit(result, device=device)
