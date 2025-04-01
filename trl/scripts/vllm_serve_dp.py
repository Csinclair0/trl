#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import multiprocessing as mp
import os
import socket
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist

from trl import TrlParser
from trl.import_utils import is_vllm_available


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.utils import get_open_port


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method.
mp.set_start_method("spawn", force=True)


@dataclass
class ScriptArguments:
    r"""
    Arguments for the data-parallel vllm-serve script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use per data parallel instance.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel instances to create. Each will run on a separate set of GPUs.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the first server on. If multiple DP instances are used, ports will be incremented.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM.
        master_addr (`str`, *optional*, defaults to `"127.0.0.1"`):
            Master address for data parallel coordination.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use per data parallel instance."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel instances to create. Each will run on a separate set of GPUs."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the first server on. If multiple DP instances are used, ports will be incremented."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache. Higher values will increase the KV cache size."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`gpu_memory_utilization`, leading to a reduced KV cache size."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    master_addr: str = field(
        default="127.0.0.1",
        metadata={"help": "Master address for data parallel coordination."},
    )


def get_free_port():
    """Get a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_dp_instance(
    dp_rank: int, 
    dp_size: int, 
    master_addr: str,
    master_port: int,
    gpus_per_dp_rank: int,
    model: str,
    revision: Optional[str],
    tensor_parallel_size: int,
    host: str,
    port: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: Optional[int],
    enable_prefix_caching: Optional[bool],
):
    """
    Run a data parallel instance of vLLM server.
    
    Args:
        dp_rank: Data parallel rank of this instance
        dp_size: Total number of data parallel instances
        master_addr: Master address for coordination
        master_port: Master port for coordination
        gpus_per_dp_rank: Number of GPUs per data parallel rank
        model: Model name or path
        revision: Model revision
        tensor_parallel_size: Tensor parallel size within this instance
        host: Host to bind the server to
        port: Port to run the server on
        gpu_memory_utilization: GPU memory utilization ratio
        dtype: Data type for computation
        max_model_len: Maximum model length
        enable_prefix_caching: Whether to enable prefix caching
    """
    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")
    
    # Set environment variables for data parallel coordination
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_ADDR"] = master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    
    # Set CUDA_VISIBLE_DEVICES to limit GPUs for this instance
    start_gpu = dp_rank * gpus_per_dp_rank
    end_gpu = start_gpu + tensor_parallel_size
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(start_gpu, end_gpu))
    
    logger.info(f"Starting DP instance {dp_rank}/{dp_size} on GPUs {start_gpu}-{end_gpu-1} on port {port}")
    
    # Create the LLM instance
    llm = LLM(
        model=model,
        revision=revision,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    
    # Now we need to start a FastAPI server for this instance
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI()
    
    # Define the health check endpoint
    @app.get("/health/")
    async def health():
        """Health check endpoint to verify that the server is running."""
        return {"status": "ok", "dp_rank": dp_rank, "dp_size": dp_size}
    
    # Define the generation endpoint
    class GenerateRequest(BaseModel):
        prompts: List[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
    
    class GenerateResponse(BaseModel):
        completion_ids: List[List[int]]
        dp_rank: int
    
    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.
        """
        # Set up guided decoding if enabled
        guided_decoding = None
        if request.guided_decoding_regex is not None:
            from vllm.sampling_params import GuidedDecodingParams
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        
        # Generate completions
        all_outputs = llm.generate(request.prompts, sampling_params=sampling_params)
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        
        return {"completion_ids": completion_ids, "dp_rank": dp_rank}
    
    # Run the server
    uvicorn.run(app, host=host, port=port)


def main(script_args: ScriptArguments):
    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")
    
    # Calculate total GPUs needed and validate
    total_gpus_needed = script_args.tensor_parallel_size * script_args.data_parallel_size
    cuda_device_count = torch.cuda.device_count()
    if total_gpus_needed > cuda_device_count:
        raise ValueError(
            f"Not enough GPUs available. Need {total_gpus_needed} GPUs "
            f"({script_args.tensor_parallel_size} * {script_args.data_parallel_size}), "
            f"but only {cuda_device_count} are available."
        )
    
    # Set up master port for coordination
    master_port = get_free_port() if is_vllm_available() else get_open_port()
    
    # Spawn processes for each data parallel instance
    processes = []
    for dp_rank in range(script_args.data_parallel_size):
        # Each DP instance gets its own port
        instance_port = script_args.port + dp_rank
        
        # Launch process
        p = mp.Process(
            target=run_dp_instance,
            args=(
                dp_rank,
                script_args.data_parallel_size,
                script_args.master_addr,
                master_port,
                script_args.tensor_parallel_size,
                script_args.model,
                script_args.revision,
                script_args.tensor_parallel_size,
                script_args.host,
                instance_port,
                script_args.gpu_memory_utilization,
                script_args.dtype,
                script_args.max_model_len,
                script_args.enable_prefix_caching,
            ),
        )
        p.start()
        processes.append(p)
        
        logger.info(f"Started DP instance {dp_rank} on port {instance_port}")
    
    # Print information for connecting
    logger.info(f"Started {script_args.data_parallel_size} data parallel vLLM instances")
    for dp_rank in range(script_args.data_parallel_size):
        instance_port = script_args.port + dp_rank
        logger.info(f"  - Instance {dp_rank}: http://{script_args.host}:{instance_port}")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve-dp", 
            help="Run data-parallel vLLM serving for faster generation",
            dataclass_types=ScriptArguments
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args) 