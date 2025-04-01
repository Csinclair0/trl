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

import atexit
import logging
import time
from typing import List, Optional, Dict, Any, Union

import torch
from torch import nn

from ..import_utils import is_requests_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


logger = logging.getLogger(__name__)


class VLLMDataParallelClient:
    """
    A client class to interact with multiple data-parallel vLLM servers.

    This class provides methods to generate completions by distributing prompts across multiple 
    vLLM server instances running in data-parallel mode. Before using it, start the multiple vLLM 
    server instances with `trl vllm-serve-dp`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address shared by all vLLM server instances.
        base_port (`int`, *optional*, defaults to `8000`):
            Base port number. Server instance with rank i will be at port base_port + i.
        dp_size (`int`, *optional*, defaults to `1`):
            Number of data parallel server instances.
        connection_timeout (`float`, *optional*, defaults to `30.0`):
            Total timeout duration in seconds to wait for all servers to be up.
        
    Examples:
        Run the vLLM data-parallel server with the model `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`:

        ```
        $ trl vllm-serve-dp --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tensor_parallel_size 2 --data_parallel_size 4
        ```

        Use the client to distribute and generate completions:

        ```python
        >>> from trl.extras.vllm_dp_client import VLLMDataParallelClient
        >>> client = VLLMDataParallelClient(dp_size=4)
        >>> client.generate(["Hello, AI!", "Tell me a joke", "Explain quantum physics", "Write a poem"])
        ```
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        base_port: int = 8000,
        dp_size: int = 1,
        connection_timeout: float = 30.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")

        self.host = host
        self.base_port = base_port
        self.dp_size = dp_size
        self.sessions = [requests.Session() for _ in range(dp_size)]
        
        # Check that all servers are up
        self.check_servers(connection_timeout)
        
        # Register cleanup
        atexit.register(self.close_sessions)

    def check_servers(self, total_timeout: float = 30.0, retry_interval: float = 2.0):
        """
        Check that all server instances are available and ready.
        
        Args:
            total_timeout (`float`, *optional*, defaults to `30.0`):
                Total timeout duration in seconds to wait for all servers.
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
        """
        start_time = time.time()
        available_servers = 0
        
        while available_servers < self.dp_size:
            for dp_rank in range(self.dp_size):
                port = self.base_port + dp_rank
                url = f"http://{self.host}:{port}/health/"
                
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        available_servers += 1
                        logger.info(f"Server {dp_rank} at port {port} is up!")
                except requests.exceptions.RequestException:
                    pass  # Server not ready yet
            
            # Check if all servers are up
            if available_servers >= self.dp_size:
                logger.info(f"All {self.dp_size} servers are up!")
                return
            
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= total_timeout:
                raise ConnectionError(
                    f"Not all vLLM servers are up after {total_timeout} seconds. "
                    f"Only {available_servers}/{self.dp_size} servers are available. "
                    f"Make sure all servers are running with `trl vllm-serve-dp`."
                )
            
            # Reset count and retry
            available_servers = 0
            logger.info(f"Not all servers are up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def split_requests(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Split a list of prompts into separate requests for each server instance.
        
        Args:
            prompts (`List[str]`):
                List of text prompts to be distributed across server instances.
            **kwargs:
                Additional arguments to be included in each request.
                
        Returns:
            `List[Dict[str, Any]]`:
                List of request payloads, one for each server instance.
        """
        # Determine how many prompts to send to each server
        prompts_per_server = len(prompts) // self.dp_size
        remainder = len(prompts) % self.dp_size
        
        # Create request payloads
        requests = []
        start_idx = 0
        
        for dp_rank in range(self.dp_size):
            # Calculate how many prompts this server gets
            count = prompts_per_server + (1 if dp_rank < remainder else 0)
            end_idx = start_idx + count
            
            # Create the request payload
            if start_idx < end_idx:
                server_prompts = prompts[start_idx:end_idx]
                payload = {"prompts": server_prompts, **kwargs}
                requests.append(payload)
            else:
                # If we ran out of prompts, create an empty request
                requests.append(None)
            
            start_idx = end_idx
        
        return requests

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> List[List[int]]:
        """
        Generates model completions for the provided prompts by distributing them across
        all available server instances.

        Args:
            prompts (`List[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `List[List[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        # Prepare generation parameters
        params = {
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
        }
        
        # Split prompts across servers
        request_payloads = self.split_requests(prompts, **params)
        
        # Track which prompts were sent to which server
        prompt_to_server_map = {}
        current_idx = 0
        
        for dp_rank, payload in enumerate(request_payloads):
            if payload is not None:
                server_prompts = payload["prompts"]
                for i in range(len(server_prompts)):
                    prompt_to_server_map[current_idx] = (dp_rank, i)
                    current_idx += 1
        
        # Send requests to each server in parallel
        responses = []
        for dp_rank, payload in enumerate(request_payloads):
            if payload is not None:
                port = self.base_port + dp_rank
                url = f"http://{self.host}:{port}/generate/"
                response = self.sessions[dp_rank].post(url, json=payload)
                
                if response.status_code == 200:
                    responses.append((dp_rank, response.json()))
                else:
                    raise Exception(f"Request to server {dp_rank} failed: {response.status_code}, {response.text}")
        
        # Combine results in the original order
        all_completion_ids = [None] * len(prompts)
        
        for dp_rank, response_data in responses:
            completions = response_data["completion_ids"]
            server_rank = response_data["dp_rank"]
            
            # Ensure server rank matches
            assert server_rank == dp_rank, f"Server rank mismatch: {server_rank} != {dp_rank}"
            
            # Map completions back to original prompt indices
            for original_idx, (rank, server_idx) in prompt_to_server_map.items():
                if rank == dp_rank:
                    # If this completion came from this server
                    if server_idx < len(completions):
                        all_completion_ids[original_idx] = completions[server_idx]
        
        # Ensure we have a completion for every prompt
        for i, completion in enumerate(all_completion_ids):
            if completion is None:
                raise Exception(f"No completion received for prompt at index {i}")
        
        return all_completion_ids

    def close_sessions(self):
        """
        Close all sessions when the client is no longer needed.
        """
        for session in self.sessions:
            session.close() 