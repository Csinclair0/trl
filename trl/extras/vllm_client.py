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
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
import warnings

from ..import_utils import is_requests_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup


logger = logging.getLogger(__name__)


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self, host: str = "0.0.0.0", server_port: int = 8000, group_port: int = 51216, connection_timeout: float = 0.0
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout
        self.init_communicator()
        atexit.register(self.close_communicator)  # when the client object is deleted, close the weight update group

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
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
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"http://{self.host}:{self.server_port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the tensor parallel size from the server
        url = f"http://{self.host}:{self.server_port}/get_tensor_parallel_size/"
        response = requests.get(url)
        if response.status_code == 200:
            tensor_parallel_size = response.json()["tensor_parallel_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size  # The client's rank is the last process

        # Initialize weight update group
        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")

    def update_named_param(self, name: str, weight: torch.Tensor, version: Optional[int] = None):
        """Update a named parameter in the model."""
        # AsyncLLM doesn't currently support this operation directly
        # This would need to be implemented when the API stabilizes
        import warnings
        
        # Track version for parameter updates
        if version is not None and not hasattr(self, "current_version"):
            self.current_version = version
        elif version is not None:
            self.current_version = max(self.current_version, version)
            
        # Try to access internal LLM engine for parameter updates
        if self.async_llm is not None:
            try:
                if hasattr(self.async_llm, "update_model_parameter"):
                    self.async_llm.update_model_parameter(name, weight)
                elif hasattr(self.async_llm, "_engine") and hasattr(self.async_llm._engine, "update_model_parameter"):
                    self.async_llm._engine.update_model_parameter(name, weight)
                else:
                    warnings.warn("Parameter updates not yet supported with AsyncLLM. "
                                 "Changes to model parameters won't be reflected in generation.")
            except Exception as e:
                warnings.warn(f"Failed to update parameter '{name}': {str(e)}")
        else:
            warnings.warn("Parameter updates not yet supported with AsyncLLM. "
                         "Changes to model parameters won't be reflected in generation.")

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.host}:{self.server_port}/close_communicator/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")


# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)


class VLLMAsyncClient:
    """Client for AsyncLLM from vLLM v1 API."""

    def __init__(self, async_llm: Optional[Any] = None, endpoint: Optional[str] = None):
        """
        Initialize the AsyncLLM client.
        
        Args:
            async_llm: An instance of AsyncLLM to use directly (local mode)
            endpoint: Socket address to connect to an AsyncLLM instance (remote mode)
        """
        self.async_llm = async_llm
        self.endpoint = endpoint
        self.current_version = 0  # Version tracking for parameter updates
        
        if self.async_llm is None and self.endpoint is None:
            # Try to import AsyncLLM and find the shared instance
            try:
                from vllm.v1.engine import get_shared_async_llm
                self.async_llm = get_shared_async_llm()
                if self.async_llm is None:
                    raise ImportError("No shared AsyncLLM instance found")
            except (ImportError, AttributeError):
                raise ImportError("AsyncLLM v1 API not available or no shared instance found")
    
    async def _generate_async(self, prompts, **kwargs):
        """Internal method to handle async generation."""
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Configure sampling parameters
        sampling_params = {}
        if "n" in kwargs:
            sampling_params["n"] = kwargs.pop("n")
        if "temperature" in kwargs:
            sampling_params["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            sampling_params["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            sampling_params["top_k"] = kwargs.pop("top_k")
        if "max_tokens" in kwargs:
            sampling_params["max_tokens"] = kwargs.pop("max_tokens")
        if "repetition_penalty" in kwargs:
            sampling_params["repetition_penalty"] = kwargs.pop("repetition_penalty")
            
        # Add any remaining kwargs
        sampling_params.update(kwargs)
        
        # Call AsyncLLM directly
        if self.async_llm is not None:
            results = await self.async_llm.generate(prompts, sampling_params=sampling_params)
            return results
        
        # TODO: Implement remote endpoint support when vLLM v1 API is more stable
        raise NotImplementedError("Remote AsyncLLM endpoint not yet supported")

    def generate(self, prompts, **kwargs):
        """
        Generate completions for prompts.
        
        Note: This is a synchronous wrapper around the async API.
        For TRL integration, we need a synchronous interface.
        """
        import asyncio
        
        # Create and run the async event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        results = loop.run_until_complete(self._generate_async(prompts, **kwargs))
        
        # Extract token IDs from results
        if isinstance(prompts, str):
            # Single prompt - return a list of completions
            return [r.outputs[0].token_ids for r in results]
        else:
            # Multiple prompts - return a list of lists
            return [[o.token_ids for o in r.outputs] for r in results]
    
    def update_named_param(self, name: str, weight: torch.Tensor, version: Optional[int] = None):
        """Update a named parameter in the model."""
        import warnings
        
        # Track version for parameter updates
        if version is not None:
            self.current_version = max(self.current_version, version)
            
        # Try to access internal AsyncLLM engine for parameter updates
        if self.async_llm is not None:
            try:
                if hasattr(self.async_llm, "update_model_parameter"):
                    self.async_llm.update_model_parameter(name, weight)
                elif hasattr(self.async_llm, "_engine") and hasattr(self.async_llm._engine, "update_model_parameter"):
                    self.async_llm._engine.update_model_parameter(name, weight)
                else:
                    warnings.warn("Parameter updates not yet supported with this AsyncLLM")
            except Exception as e:
                warnings.warn(f"Failed to update parameter '{name}': {str(e)}")
        else:
            warnings.warn("Parameter updates not supported with remote AsyncLLM endpoint")
    
    def reset_prefix_cache(self):
        """Reset the KV cache in the model."""
        import asyncio
        import warnings
        
        if self.async_llm is not None:
            # Try to access the internal LLM engine to reset cache
            try:
                # Create and run the async event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                async def reset_cache():
                    if hasattr(self.async_llm, "reset_cache"):
                        await self.async_llm.reset_cache()
                    elif hasattr(self.async_llm, "_engine") and hasattr(self.async_llm._engine, "reset_cache"):
                        await self.async_llm._engine.reset_cache()
                
                # Reset cache
                loop.run_until_complete(reset_cache())
            except Exception as e:
                warnings.warn(f"Failed to reset prefix cache: {str(e)}")
        # No action needed for remote AsyncLLM as cache is managed internally
