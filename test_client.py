#!/usr/bin/env python3
"""
Test client for CogniVault gRPC server.
"""
import asyncio
import argparse
import json
import logging
import uuid

import grpc
from grpc import aio

from proto import cognivault_pb2
from proto import cognivault_pb2_grpc


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_ping(channel: aio.Channel, message: str = "Hello"):
    """Test the Ping RPC."""
    stub = cognivault_pb2_grpc.CogniVaultServiceStub(channel)
    
    request = cognivault_pb2.PingRequest(message=message)
    logger.info(f"Sending ping request: {message}")
    
    try:
        response = await stub.Ping(request)
        logger.info(f"Ping response: {response.response}, timestamp: {response.timestamp}")
        return response
    except grpc.RpcError as e:
        logger.error(f"Ping failed with status {e.code()}: {e.details()}")
        raise


async def test_process_data(channel: aio.Channel, data: str, metadata: dict = None):
    """Test the ProcessData RPC."""
    stub = cognivault_pb2_grpc.CogniVaultServiceStub(channel)
    
    request = cognivault_pb2.ProcessDataRequest(
        data=data,
        metadata=metadata or {}
    )
    logger.info(f"Sending process data request: {len(data)} bytes")
    
    try:
        response = await stub.ProcessData(request)
        # Parse JSON result
        try:
            result_dict = json.loads(response.result)
            if result_dict.get("success", False):
                logger.info(f"Process data success: {json.dumps(result_dict, indent=2)}")
            else:
                logger.error(f"Process data failed: {result_dict.get('error', 'Unknown error')}")
        except json.JSONDecodeError:
            logger.warning(f"Response is not valid JSON: {response.result}")
        return response
    except grpc.RpcError as e:
        logger.error(f"Process data failed with status {e.code()}: {e.details()}")
        raise


async def test_process_input(channel: aio.Channel, chat_history: cognivault_pb2.ChatHistory):
    """Test the ProcessInput RPC (model input processor)."""
    stub = cognivault_pb2_grpc.CogniVaultServiceStub(channel)
    
    request = cognivault_pb2.ProcessInputRequest(chat_history=chat_history)
    logger.info(f"Sending process input request with {len(chat_history.messages)} messages")
    
    try:
        response = await stub.ProcessInput(request)
        if response.success:
            logger.info("Process input success!")
            batch_input = response.batch_input
            if batch_input:
                logger.info(f"Batch input received:")
                if batch_input.input_ids:
                    logger.info(f"  - input_ids: shape {list(batch_input.input_ids.shape)}")
                if batch_input.label_ids:
                    logger.info(f"  - label_ids: shape {list(batch_input.label_ids.shape)}")
                if batch_input.attention_mask:
                    logger.info(f"  - attention_mask: shape {list(batch_input.attention_mask.shape)}")
                if batch_input.audio_in_ids:
                    logger.info(f"  - audio_in_ids: shape {list(batch_input.audio_in_ids.shape)}")
                if batch_input.audio_out_ids:
                    logger.info(f"  - audio_out_ids: shape {list(batch_input.audio_out_ids.shape)}")
                if batch_input.audio_features:
                    logger.info(f"  - audio_features: shape {list(batch_input.audio_features.shape)}")
                if batch_input.reward:
                    logger.info(f"  - reward: {batch_input.reward}")
                # Log data sizes for debugging
                if batch_input.input_ids and batch_input.input_ids.int64_data:
                    logger.info(f"  - input_ids data size: {len(batch_input.input_ids.int64_data)}")
                if batch_input.label_ids and batch_input.label_ids.int64_data:
                    logger.info(f"  - label_ids data size: {len(batch_input.label_ids.int64_data)}")
            else:
                logger.warning("Batch input is None")
        else:
            logger.error(f"Process input failed: {response.error_message}")
        return response
    except grpc.RpcError as e:
        logger.error(f"Process input failed with status {e.code()}: {e.details()}")
        raise


async def run_tests(host: str = "localhost", port: int = 50051):
    """Run all tests."""
    address = f"{host}:{port}"
    logger.info(f"Connecting to server at {address}")
    
    async with aio.insecure_channel(address) as channel:
        # Test ping
        logger.info("\n=== Testing Ping ===")
        await test_ping(channel, "Test ping message")
        
        # Test process data
        logger.info("\n=== Testing ProcessData ===")
        test_data_dict = {
            "id": str(uuid.uuid4()),
            "messages": [
                {
                    "role": "user",
                    "text_content": {
                        "text": "Hello, how are you?",
                        "type": "text"
                    }
                },
                {
                    "role": "assistant",
                    "text_content": {
                        "text": "I'm doing well, thank you for asking!",
                        "type": "text"
                    }
                },
                {
                    "role": "user",
                    "text_content": {
                        "text": "Can you help me with a question?",
                        "type": "text"
                    }
                }
            ],
            "metadata": {
                "source": "test_client",
                "type": "text_chat"
            }
        }
        test_data = json.dumps(test_data_dict)
        test_metadata = {
            "source": "test_client",
            "type": "text",
            "timestamp": "1234567890"
        }
        await test_process_data(channel, test_data, test_metadata)
        
        # Test process data with empty metadata
        logger.info("\n=== Testing ProcessData (no metadata) ===")
        test_data_dict_simple = {
            "id": str(uuid.uuid4()),
            "messages": [
                {
                    "role": "user",
                    "text_content": {
                        "text": "Another test message",
                        "type": "text"
                    }
                }
            ]
        }
        await test_process_data(channel, json.dumps(test_data_dict_simple))
        
        # Test ProcessInput with text messages
        logger.info("\n=== Testing ProcessInput (text messages) ===")
        chat_history_text = cognivault_pb2.ChatHistory(
            id=str(uuid.uuid4()),
            messages=[
                cognivault_pb2.Message(
                    role="user",
                    text_content=cognivault_pb2.TextContent(text="Hello, how are you?")
                ),
                cognivault_pb2.Message(
                    role="assistant",
                    text_content=cognivault_pb2.TextContent(text="I'm doing well, thank you for asking!")
                ),
                cognivault_pb2.Message(
                    role="user",
                    text_content=cognivault_pb2.TextContent(text="Can you help me with a question?")
                ),
            ],
            metadata={"source": "test_client", "type": "text_chat"}
        )
        await test_process_input(channel, chat_history_text)
        
        # Test ProcessInput with audio messages
        logger.info("\n=== Testing ProcessInput (audio messages) ===")
        chat_history_audio = cognivault_pb2.ChatHistory(
            id=str(uuid.uuid4()),
            messages=[
                cognivault_pb2.Message(
                    role="user",
                    audio_content=cognivault_pb2.AudioContent(
                        audio_url="placeholder",
                        raw_audio=""  # Empty for placeholder
                    )
                ),
                cognivault_pb2.Message(
                    role="assistant",
                    audio_content=cognivault_pb2.AudioContent(
                        audio_url="placeholder",
                        raw_audio=""
                    )
                ),
            ],
            metadata={"source": "test_client", "type": "audio_chat"}
        )
        await test_process_input(channel, chat_history_audio)
        
        # Test ProcessInput with mixed text and audio
        logger.info("\n=== Testing ProcessInput (mixed text and audio) ===")
        chat_history_mixed = cognivault_pb2.ChatHistory(
            id=str(uuid.uuid4()),
            messages=[
                cognivault_pb2.Message(
                    role="user",
                    text_content=cognivault_pb2.TextContent(text="Please listen to this audio")
                ),
                cognivault_pb2.Message(
                    role="user",
                    audio_content=cognivault_pb2.AudioContent(
                        audio_url="placeholder",
                        raw_audio=""
                    )
                ),
                cognivault_pb2.Message(
                    role="assistant",
                    text_content=cognivault_pb2.TextContent(text="I've processed the audio.")
                ),
            ],
            metadata={"source": "test_client", "type": "mixed_chat"}
        )
        await test_process_input(channel, chat_history_mixed)
        
        # Test ProcessInput with recipient (assistant message)
        logger.info("\n=== Testing ProcessInput (with recipient) ===")
        chat_history_recipient = cognivault_pb2.ChatHistory(
            id=str(uuid.uuid4()),
            messages=[
                cognivault_pb2.Message(
                    role="user",
                    text_content=cognivault_pb2.TextContent(text="Who should handle this?")
                ),
                cognivault_pb2.Message(
                    role="assistant",
                    text_content=cognivault_pb2.TextContent(text="I'll handle this request."),
                    recipient="specialist"
                ),
            ],
            metadata={"source": "test_client", "type": "recipient_test"}
        )
        await test_process_input(channel, chat_history_recipient)
        
        logger.info("\n=== All tests completed ===")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test client for CogniVault gRPC server')
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Server host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=50051,
        help='Server port (default: 50051)'
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(run_tests(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Test client stopped")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

