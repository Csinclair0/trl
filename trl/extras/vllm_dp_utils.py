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

import os
from typing import Dict, Optional, Tuple


def setup_vllm_data_parallel_env(
    dp_rank: int = 0,
    dp_size: int = 1,
    master_ip: str = "127.0.0.1",
    master_port: int = 51217,
) -> None:
    """
    Set up the environment variables for vLLM data parallelism.

    Args:
        dp_rank (`int`, *optional*, defaults to `0`):
            The rank of this process in the data parallel group.
        dp_size (`int`, *optional*, defaults to `1`):
            The total number of processes in the data parallel group.
        master_ip (`str`, *optional*, defaults to `"127.0.0.1"`):
            The IP address of the master node.
        master_port (`int`, *optional*, defaults to `51217`):
            The port number of the master node.
    """
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)


def get_vllm_data_parallel_info() -> Dict[str, str]:
    """
    Get information about the current vLLM data parallelism configuration.

    Returns:
        `Dict[str, str]`:
            A dictionary containing the current data parallelism configuration, including
            the rank, size, master IP, and master port.
    """
    return {
        "rank": os.environ.get("VLLM_DP_RANK", "0"),
        "size": os.environ.get("VLLM_DP_SIZE", "1"),
        "master_ip": os.environ.get("VLLM_DP_MASTER_IP", "127.0.0.1"),
        "master_port": os.environ.get("VLLM_DP_MASTER_PORT", "51217"),
    }


def get_client_port_for_rank(base_port: int = 8000, rank: Optional[int] = None) -> int:
    """
    Get the port number for a specific vLLM server based on rank.

    When running multiple vLLM servers for data parallelism, each server needs to run on a 
    different port. This function calculates the port based on the rank and a base port.

    Args:
        base_port (`int`, *optional*, defaults to `8000`):
            The base port number for the vLLM servers.
        rank (`int` or `None`, *optional*, defaults to `None`):
            The rank of the process. If None, the rank is determined from the environment
            variable VLLM_DP_RANK.

    Returns:
        `int`:
            The port number for the vLLM server with the specified rank.
    """
    if rank is None:
        rank = int(os.environ.get("VLLM_DP_RANK", "0"))
    return base_port + rank


def get_shard_info(total_items: int) -> Tuple[int, int]:
    """
    Calculate the start and end indices for the current rank's shard of data.

    Args:
        total_items (`int`):
            The total number of items to be distributed.

    Returns:
        `Tuple[int, int]`:
            A tuple containing the start and end indices for the current rank's shard of data.
    """
    dp_rank = int(os.environ.get("VLLM_DP_RANK", "0"))
    dp_size = int(os.environ.get("VLLM_DP_SIZE", "1"))
    
    items_per_rank = total_items // dp_size
    if items_per_rank == 0:
        # If we have more ranks than items, assign at least one item to each rank
        if dp_rank < total_items:
            start_idx = dp_rank
            end_idx = dp_rank + 1
        else:
            # No items for this rank
            start_idx = 0
            end_idx = 0
    else:
        start_idx = dp_rank * items_per_rank
        end_idx = start_idx + items_per_rank
        # If there are remaining items, distribute them among the first few ranks
        remaining = total_items % dp_size
        if dp_rank < remaining:
            start_idx += dp_rank
            end_idx += dp_rank + 1
        else:
            start_idx += remaining
            end_idx += remaining
            
    return start_idx, end_idx 