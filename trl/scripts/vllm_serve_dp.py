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
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Process
from typing import Optional

from trl import TrlParser
from trl.import_utils import is_vllm_available
from trl.scripts.vllm_serve import ScriptArguments as BaseScriptArguments
from trl.scripts.vllm_serve import main as vllm_serve_main


@dataclass
class DataParallelScriptArguments(BaseScriptArguments):
    r"""
    Arguments for the data parallel vLLM server script.

    Args:
        dp_size (`int`, *optional*, defaults to `2`):
            Number of data parallel workers to use.
        dp_port (`int`, *optional*, defaults to `51217`):
            Port to use for data parallel communication.
        node_size (`int`, *optional*, defaults to `1`):
            Total number of nodes in the cluster.
        node_rank (`int`, *optional*, defaults to `0`):
            Rank of the current node in the cluster.
        master_addr (`str`, *optional*, defaults to `"127.0.0.1"`):
            IP address of the master node.
        master_port (`int`, *optional*, defaults to `51217`):
            Port number of the master node.
    """

    dp_size: int = field(
        default=2,
        metadata={"help": "Number of data parallel workers to use."},
    )
    dp_port: int = field(
        default=51217,
        metadata={"help": "Port to use for data parallel communication."},
    )
    node_size: int = field(
        default=1,
        metadata={"help": "Total number of nodes in the cluster."},
    )
    node_rank: int = field(
        default=0,
        metadata={"help": "Rank of the current node in the cluster."},
    )
    master_addr: str = field(
        default="127.0.0.1",
        metadata={"help": "IP address of the master node."},
    )
    master_port: int = field(
        default=51217,
        metadata={"help": "Port number of the master node."},
    )


def run_vllm_server(args, dp_rank, global_dp_rank, port):
    """Run a vLLM server with specific data parallel rank and port."""
    # Set environment variables for data parallelism
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = args.master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_port)

    # Create a new args object with the updated port
    vllm_args = BaseScriptArguments(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        host=args.host,
        port=port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # Run the vLLM server
    vllm_serve_main(vllm_args)


def main(args: DataParallelScriptArguments):
    """Main function to launch multiple vLLM servers for data parallel inference."""
    if not is_vllm_available():
        raise ImportError("vLLM is required to run data parallel vLLM servers. Please install it with `pip install vllm`")

    dp_size = args.dp_size
    node_size = args.node_size
    node_rank = args.node_rank

    # Validate arguments
    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    # Calculate the range of global ranks for this node
    start_rank = node_rank * dp_per_node
    end_rank = start_rank + dp_per_node

    # Start one process per data parallel worker on this node
    processes = []
    for local_dp_rank, global_dp_rank in enumerate(range(start_rank, end_rank)):
        # Each server needs its own port - calculate it based on the base port and rank
        server_port = args.port + global_dp_rank

        proc = Process(
            target=run_vllm_server,
            args=(args, local_dp_rank, global_dp_rank, server_port),
            daemon=True,
        )
        proc.start()
        processes.append(proc)
        print(f"Started vLLM server with DP rank {global_dp_rank} on port {server_port}")

    # Give the servers time to start up
    time.sleep(5)
    print(f"All {dp_per_node} vLLM servers started on this node")

    # Keep the servers running
    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        print("Shutting down vLLM servers...")
        for proc in processes:
            proc.terminate()
        sys.exit(0)


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create or add an argument parser for the vLLM data parallel server script."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve-dp", help="Run multiple vLLM servers for data parallel inference", dataclass_types=DataParallelScriptArguments
        )
    else:
        parser = TrlParser(dataclass_types=DataParallelScriptArguments)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args) 