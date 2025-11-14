#!/usr/bin/env python3
"""
Async gRPC server for CogniVault service.
"""
import asyncio
import argparse
import json
import logging
import time
import uuid
from typing import Dict, Optional

import grpc
from grpc import aio
import torch
import numpy as np

from proto import cognivault_pb2
from proto import cognivault_pb2_grpc

from src.input.chat_history import ChatHistory
from src.input.message import Message
from src.input.message_content import AudioContent, TextContent
from src.engine.engine import AudioEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CogniVaultServicer(cognivault_pb2_grpc.CogniVaultServiceServicer):
    """Async implementation of CogniVaultService."""
    
    def __init__(self):
        """
        Initialize the servicer.
        
        Args:
            input_processor: Optional InputProcessor instance. If None, will need to be set later.
        """
        self.engine = AudioEngine()
    
    async def Ping(self, request: cognivault_pb2.PingRequest, context: aio.ServicerContext) -> cognivault_pb2.PingResponse:
        """Simple ping method to test connectivity."""
        logger.info(f"Received ping request: {request.message}")
        response = cognivault_pb2.PingResponse(
            response=f"Pong: {request.message}",
            timestamp=int(time.time())
        )
        return response
    
    async def ProcessData(self, request: cognivault_pb2.ProcessDataRequest, context: aio.ServicerContext) -> cognivault_pb2.ProcessDataResponse:
        """Process data with metadata."""
        logger.info(f"Processing data: {len(request.data)} bytes, metadata: {dict(request.metadata)}")
        
        try:
            # Call the cognivault process data function
            result_json = await self._process_cognivault_data(request.data, dict(request.metadata))
            
            response = cognivault_pb2.ProcessDataResponse(result=result_json)
            logger.info("Data processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            # Return error as JSON string
            error_result = json.dumps({
                "error": str(e),
                "success": False
            })
            response = cognivault_pb2.ProcessDataResponse(result=error_result)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return response
    
    async def _process_cognivault_data(self, data: str, metadata: Dict[str, str]) -> str:
        """
        Process data using cognivault processing logic.
        
        Parses JSON data string into ChatHistory, processes it using InputProcessor,
        and returns a JSON string with the processing results.
        
        Args:
            data: JSON string representing a ChatHistory object
            metadata: Additional metadata dictionary
            
        Returns:
            JSON string containing processing results
        """
        
        # Parse JSON data into ChatHistory
        try:
            data_dict = json.loads(data)
            chat_history = self._json_to_chat_history(data_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse ChatHistory from JSON: {str(e)}")
        
        # Process using InputProcessor (CPU-bound, run in executor)
        loop = asyncio.get_event_loop()
        batch_input = await loop.run_in_executor(None, self.engine.generate, chat_history, 1024)
        
        # Build result dictionary
        result = {
            "success": True,
            "message_count": len(chat_history.messages),
            "shapes": {}
        }
        
        # Extract tensor shapes
        if hasattr(batch_input, 'input_ids') and batch_input.input_ids is not None:
            result["shapes"]["input_ids"] = list(batch_input.input_ids.shape)
        
        if hasattr(batch_input, 'label_ids') and batch_input.label_ids is not None:
            result["shapes"]["label_ids"] = list(batch_input.label_ids.shape)
        
        if hasattr(batch_input, 'audio_in_ids') and batch_input.audio_in_ids is not None:
            result["shapes"]["audio_in_ids"] = list(batch_input.audio_in_ids.shape)
        
        if hasattr(batch_input, 'audio_out_ids') and batch_input.audio_out_ids is not None:
            result["shapes"]["audio_out_ids"] = list(batch_input.audio_out_ids.shape)
        
        if hasattr(batch_input, 'audio_features') and batch_input.audio_features is not None:
            result["shapes"]["audio_features"] = list(batch_input.audio_features.shape)
        
        # Add metadata if present
        if metadata:
            result["metadata"] = metadata
        
        # Return as JSON string
        return json.dumps(result)
    
    def _json_to_chat_history(self, data_dict: dict) -> ChatHistory:
        """Convert JSON dictionary to ChatHistory object."""
        # Extract chat ID
        chat_id = uuid.UUID(data_dict.get('id', str(uuid.uuid4())))
        
        # Extract messages
        messages = []
        for msg_dict in data_dict.get('messages', []):
            # Determine content type
            content = None
            if 'text_content' in msg_dict:
                text_data = msg_dict['text_content']
                content = TextContent(
                    text=text_data.get('text', ''),
                    type=text_data.get('type', 'text')
                )
            elif 'audio_content' in msg_dict:
                audio_data = msg_dict['audio_content']
                content = AudioContent(
                    audio_url=audio_data.get('audio_url', ''),
                    raw_audio=audio_data.get('raw_audio'),
                    type=audio_data.get('type', 'audio')
                )
            else:
                # Fallback: try to infer from 'content' field
                if 'content' in msg_dict:
                    if isinstance(msg_dict['content'], str):
                        # Assume text content
                        content = TextContent(text=msg_dict['content'])
                    else:
                        raise ValueError("Cannot determine content type from message")
                else:
                    raise ValueError("Message must have either text_content or audio_content")
            
            message = Message(
                role=msg_dict.get('role', 'user'),
                content=content,
                recipient=msg_dict.get('recipient')
            )
            messages.append(message)
        
        # Extract metadata
        metadata = data_dict.get('metadata')
        
        return ChatHistory(id=chat_id, messages=messages, metadata=metadata)
    
    async def ProcessInput(self, request: cognivault_pb2.ProcessInputRequest, context: aio.ServicerContext) -> cognivault_pb2.ProcessInputResponse:
        """Process chat history using InputProcessor and return ModelBatchInput."""
        logger.info(f"Processing input request with {len(request.chat_history.messages)} messages")
        
        try:
            # Convert proto ChatHistory to Python ChatHistory
            chat_history = self._proto_to_chat_history(request.chat_history)
            
            # Process using InputProcessor (this is CPU-bound, so we run in executor)
            loop = asyncio.get_event_loop()
            batch_input = await loop.run_in_executor(None, self.engine.generate, chat_history, 1024)
            
            # Convert ModelBatchInput to proto response
            proto_batch_input = self._model_batch_input_to_proto(batch_input)
            
            response = cognivault_pb2.ProcessInputResponse(
                success=True,
                error_message="",
                batch_input=proto_batch_input
            )
            logger.info("Input processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            response = cognivault_pb2.ProcessInputResponse(
                success=False,
                error_message=str(e),
                batch_input=None
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return response
    
    def _proto_to_chat_history(self, proto_chat: cognivault_pb2.ChatHistory) -> ChatHistory:
        """Convert proto ChatHistory to Python ChatHistory."""
        chat_id = uuid.UUID(proto_chat.id) if proto_chat.id else uuid.uuid4()
        
        messages = []
        for proto_msg in proto_chat.messages:
            # Determine content type
            if proto_msg.HasField('text_content'):
                content = TextContent(text=proto_msg.text_content.text)
            elif proto_msg.HasField('audio_content'):
                audio_proto = proto_msg.audio_content
                content = AudioContent(
                    audio_url=audio_proto.audio_url,
                    raw_audio=audio_proto.raw_audio if audio_proto.raw_audio else None,
                    type="audio"
                )
            else:
                raise ValueError("Message must have either text_content or audio_content")
            
            message = Message(
                role=proto_msg.role,
                content=content,
                recipient=proto_msg.recipient if proto_msg.recipient else None
            )
            messages.append(message)
        
        metadata = dict(proto_chat.metadata) if proto_chat.metadata else None
        
        return ChatHistory(id=chat_id, messages=messages, metadata=metadata)
    
    def _tensor_to_proto(self, tensor: Optional[torch.Tensor]) -> Optional[cognivault_pb2.TensorData]:
        """Convert a torch.Tensor to proto TensorData."""
        if tensor is None:
            return None
        
        tensor_np = tensor.cpu().detach().numpy()
        shape = list(tensor_np.shape)
        dtype_str = str(tensor_np.dtype)
        
        proto_tensor = cognivault_pb2.TensorData()
        proto_tensor.shape.extend(shape)
        proto_tensor.dtype = dtype_str
        
        # Convert to appropriate data type
        if 'int' in dtype_str:
            proto_tensor.int64_data.extend(tensor_np.flatten().astype(np.int64).tolist())
        elif 'float' in dtype_str:
            proto_tensor.float_data.extend(tensor_np.flatten().astype(np.float32).tolist())
        else:
            # Fallback to raw bytes
            proto_tensor.raw_data = tensor_np.tobytes()
        
        return proto_tensor
    
    def _model_batch_input_to_proto(self, batch_input) -> cognivault_pb2.ModelBatchInput:
        """Convert ModelBatchInput to proto ModelBatchInput."""
        proto_batch = cognivault_pb2.ModelBatchInput()
        
        # Convert each tensor field (only set if tensor exists and is not None)
        tensor_fields = [
            ('input_ids', 'input_ids'),
            ('attention_mask', 'attention_mask'),
            ('audio_features', 'audio_features'),
            ('audio_feature_attention_mask', 'audio_feature_attention_mask'),
            ('audio_out_ids', 'audio_out_ids'),
            ('audio_out_ids_start', 'audio_out_ids_start'),
            ('audio_out_ids_start_group_loc', 'audio_out_ids_start_group_loc'),
            ('audio_in_ids', 'audio_in_ids'),
            ('audio_in_ids_start', 'audio_in_ids_start'),
            ('label_ids', 'label_ids'),
            ('label_audio_ids', 'label_audio_ids'),
        ]
        
        for attr_name, proto_field_name in tensor_fields:
            if hasattr(batch_input, attr_name):
                tensor = getattr(batch_input, attr_name)
                if tensor is not None:
                    proto_tensor = self._tensor_to_proto(tensor)
                    if proto_tensor is not None:
                        getattr(proto_batch, proto_field_name).CopyFrom(proto_tensor)
        
        if hasattr(batch_input, 'reward') and batch_input.reward is not None:
            # Handle tensor rewards (may be empty)
            if isinstance(batch_input.reward, torch.Tensor):
                if batch_input.reward.numel() > 0:
                    # If tensor has elements, use the first one (or mean if multiple)
                    proto_batch.reward = float(batch_input.reward[0] if batch_input.reward.dim() > 0 else batch_input.reward.item())
                # If tensor is empty, skip setting reward
            else:
                # If it's already a scalar, convert directly
                proto_batch.reward = float(batch_input.reward)
        
        return proto_batch


async def serve(port: int = 50051):
    """Start the async gRPC server."""
    server = aio.server()
    
    # Create async servicer instance
    servicer = CogniVaultServicer()
    
    # Add async method handlers directly
    # For async gRPC, we use method_handlers_generic_handler with handler functions
    rpc_method_handlers = {
        'Ping': grpc.unary_unary_rpc_method_handler(
            servicer.Ping,
            request_deserializer=cognivault_pb2.PingRequest.FromString,
            response_serializer=cognivault_pb2.PingResponse.SerializeToString,
        ),
        'ProcessData': grpc.unary_unary_rpc_method_handler(
            servicer.ProcessData,
            request_deserializer=cognivault_pb2.ProcessDataRequest.FromString,
            response_serializer=cognivault_pb2.ProcessDataResponse.SerializeToString,
        ),
        'ProcessInput': grpc.unary_unary_rpc_method_handler(
            servicer.ProcessInput,
            request_deserializer=cognivault_pb2.ProcessInputRequest.FromString,
            response_serializer=cognivault_pb2.ProcessInputResponse.SerializeToString,
        ),
    }
    
    generic_handler = grpc.method_handlers_generic_handler(
        'cognivault.CogniVaultService', rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    
    # Listen on port
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting CogniVault gRPC server on {listen_addr}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop(5)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CogniVault async gRPC server')
    parser.add_argument(
        '--port',
        type=int,
        default=50051,
        help='Port to listen on (default: 50051)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debugpy debugger on port 5678'
    )
    parser.add_argument(
        '--debug-port',
        type=int,
        default=5678,
        help='Port for debugpy debugger (default: 5678)'
    )
    args = parser.parse_args()
    
    # Start debugpy if debug mode is enabled
    if args.debug:
        import debugpy
        logger.info(f"Starting debugpy on port {args.debug_port}")
        debugpy.listen(("0.0.0.0", args.debug_port))
        logger.info(f"Waiting for debugger to attach on port {args.debug_port}...")
        # Uncomment the line below if you want to wait for debugger before starting
        # debugpy.wait_for_client()
    
    try:
        asyncio.run(serve(args.port))
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == '__main__':
    main()

