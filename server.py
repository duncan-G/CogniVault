#!/usr/bin/env python3
"""Inference engine gRPC server.

Exposes the AsyncLLMEngine over gRPC so clients can send inference requests
and stream responses. Supports graceful shutdown on SIGINT/SIGTERM.

Run:
  python server.py
  python server.py --host 0.0.0.0 --port 50051
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

import grpc

from inference_grpc import inference_engine_pb2 as pb2
from inference_grpc import inference_engine_pb2_grpc as pb2_grpc

from model_config import model_config
from src.engine import AsyncLLMEngine
from src.engine.servicer import InferenceEngineServicer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI & server setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser; host/port can also be set via HOST and PORT env vars."""
    p = argparse.ArgumentParser(description="Inference engine gRPC server")
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "50051")))
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return p


def enable_reflection(server: grpc.aio.Server) -> None:
    """Enable gRPC server reflection so tools (e.g. grpcurl) can discover services."""
    try:
        from grpc_reflection.v1alpha import reflection

        service_names = (
            pb2.DESCRIPTOR.services_by_name["InferenceEngine"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        logger.info("gRPC reflection enabled")
    except ImportError:
        logger.info("gRPC reflection disabled (pip install grpcio-reflection)")


async def shutdown_engine(engine: AsyncLLMEngine) -> None:
    """Call the engine's shutdown hook if present (async or sync)."""
    shutdown = getattr(engine, "shutdown", None)
    if not shutdown:
        return
    if asyncio.iscoroutinefunction(shutdown):
        await shutdown()
    else:
        shutdown()


# ---------------------------------------------------------------------------
# gRPC server lifecycle
# ---------------------------------------------------------------------------


async def serve(engine: AsyncLLMEngine, host: str, port: int) -> None:
    """Run the gRPC server until SIGINT or SIGTERM; then stop server and engine."""
    servicer = InferenceEngineServicer(engine, start_time=time.time())
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    pb2_grpc.add_InferenceEngineServicer_to_server(servicer, server)
    enable_reflection(server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    await server.start()
    logger.info("Inference gRPC server started on %s", address)

    # Block until we receive SIGINT or SIGTERM
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        logger.info("Shutting down gRPC server …")
        await server.stop(grace=5.0)
        await shutdown_engine(engine)
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args, configure logging and engine, then run the gRPC server."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Configuring engine (model=%s, cache_dir=%s) …",
        model_config.model_name_or_path,
        model_config.model_cache_dir,
    )
    engine = AsyncLLMEngine(model_config=model_config)

    try:
        asyncio.run(serve(engine, host=args.host, port=args.port))
    except Exception:
        logger.exception("Server failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
