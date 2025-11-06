#!/usr/bin/env python3
"""
Simple gRPC test client for CogniVault
"""
import grpc
from proto import cognivault_pb2
from proto import cognivault_pb2_grpc


def test_ping(channel):
    """Test Ping method"""
    stub = cognivault_pb2_grpc.CogniVaultServiceStub(channel)
    request = cognivault_pb2.PingRequest(message="Hello, CogniVault!")
    response = stub.Ping(request)
    print(f"Ping Response: {response.response}")
    print(f"Timestamp: {response.timestamp}")


def test_process_data(channel):
    """Test ProcessData method"""
    stub = cognivault_pb2_grpc.CogniVaultServiceStub(channel)
    request = cognivault_pb2.ProcessDataRequest(
        data="test data",
        metadata={"key1": "value1", "key2": "value2"}
    )
    response = stub.ProcessData(request)
    print(f"ProcessData Response:")
    print(f"  Success: {response.success}")
    print(f"  Result: {response.result}")
    print(f"  Message: {response.message}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CogniVault gRPC Test Client')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=50052, help='Server port (default: 50052)')
    args = parser.parse_args()
    
    address = f'{args.host}:{args.port}'
    print(f'Connecting to {address}...')
    
    with grpc.insecure_channel(address) as channel:
        try:
            print('\n=== Testing Ping ===')
            test_ping(channel)
            
            print('\n=== Testing ProcessData ===')
            test_process_data(channel)
            
        except grpc.RpcError as e:
            print(f'Error: {e.code()} - {e.details()}')

