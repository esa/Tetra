# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from tetra3 import cedar_detect_pb2 as tetra3_dot_cedar__detect__pb2

GRPC_GENERATED_VERSION = '1.64.1'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.65.0'
SCHEDULED_RELEASE_DATE = 'June 25, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in tetra3/cedar_detect_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class CedarDetectStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ExtractCentroids = channel.unary_unary(
                '/cedar_detect.CedarDetect/ExtractCentroids',
                request_serializer=tetra3_dot_cedar__detect__pb2.CentroidsRequest.SerializeToString,
                response_deserializer=tetra3_dot_cedar__detect__pb2.CentroidsResult.FromString,
                _registered_method=True)


class CedarDetectServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ExtractCentroids(self, request, context):
        """Returns INTERNAL error if the Image request's shared memory cannot be accessed.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CedarDetectServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ExtractCentroids': grpc.unary_unary_rpc_method_handler(
                    servicer.ExtractCentroids,
                    request_deserializer=tetra3_dot_cedar__detect__pb2.CentroidsRequest.FromString,
                    response_serializer=tetra3_dot_cedar__detect__pb2.CentroidsResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'cedar_detect.CedarDetect', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('cedar_detect.CedarDetect', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class CedarDetect(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ExtractCentroids(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/cedar_detect.CedarDetect/ExtractCentroids',
            tetra3_dot_cedar__detect__pb2.CentroidsRequest.SerializeToString,
            tetra3_dot_cedar__detect__pb2.CentroidsResult.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
