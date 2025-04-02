#!/usr/bin/env python3
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Script to launch vLLM AsyncLLM with data parallelism for TRL.

This script launches a AsyncLLM instance with data parallelism and exposes
it to be used by the TRL GRPOTrainer for parallel inference.
"""

import argparse
import os
import time
from typing import List, Optional

try:
    from vllm.v1.engine import AsyncLLM
except ImportError:
    raise ImportError(
        "vLLM v1 with AsyncLLM is not available. "
        "Please install the latest version of vLLM from source: "
        "`pip install git+https://github.com/vllm-project/vllm.git`"
    )


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM AsyncLLM with data parallelism for TRL")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--revision", type=str, default=None, help="Model revision")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Data type for model weights and activations",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable prefix caching",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=0,  # 0 means auto-detect
        help="Number of data parallel instances (0 means all available GPUs)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism per instance",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length in tokens",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Token block size for PagedAttention",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="CPU swap space size (GiB)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens in a batch",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per batch",
    )

    args = parser.parse_args()

    # Create AsyncLLM instance with data parallelism
    llm = AsyncLLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size if args.data_parallel_size > 0 else None,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        block_size=args.block_size,
        swap_space=args.swap_space,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    # Keep the process running
    print(f"AsyncLLM started with data_parallel_size={llm.data_parallel_size}")
    print(f"To use in TRL, set the following in GRPOConfig:")
    print(f"    use_vllm=True")
    print(f"    vllm_async_mode=True")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down AsyncLLM...")
        # AsyncLLM will clean up resources automatically when the process exits


if __name__ == "__main__":
    main() 