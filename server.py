#!/usr/bin/env python3
"""
gRPC Server for CogniVault
"""
import grpc
from concurrent import futures
import time
from proto import cognivault_pb2
from proto import cognivault_pb2_grpc
import asyncio

class CogniVaultServicer(cognivault_pb2_grpc.CogniVaultServiceServicer):
    """Implementation of CogniVaultService"""
    
    def Ping(self, request, context):
        """Handle Ping requests"""
        response = cognivault_pb2.PingResponse(
            response=f"Echo: {request.message}",
            timestamp=int(time.time())
        )
        return response
    
    def ProcessData(self, request, context):
        """Handle ProcessData requests"""
        # Example processing logic
        result = f"Processed: {request.data}"
        if request.metadata:
            metadata_str = ", ".join([f"{k}={v}" for k, v in request.metadata.items()])
            result += f" (metadata: {metadata_str})"
        
        response = cognivault_pb2.ProcessDataResponse(
            result=result,
            success=True,
            message="Data processed successfully"
        )
        return response


async def serve() -> None:
    server = grpc.aio.server()
    cognivault_pb2_grpc.add_CogniVaultServiceServicer_to_server(
        CogniVaultServicer(), server
    )
    server.add_insecure_port('[::]:50052')
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())