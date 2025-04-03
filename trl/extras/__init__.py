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

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "best_of_n_sampler": ["BestOfNSampler"],
    "vllm_dp_utils": [
        "setup_vllm_data_parallel_env",
        "get_vllm_data_parallel_info",
        "get_client_port_for_rank",
        "get_shard_info",
    ],
}

if TYPE_CHECKING:
    from .best_of_n_sampler import BestOfNSampler
    from .vllm_dp_utils import (
        setup_vllm_data_parallel_env,
        get_vllm_data_parallel_info,
        get_client_port_for_rank,
        get_shard_info,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

__all__ = [
    "BestOfNSampler",
    "setup_vllm_data_parallel_env",
    "get_vllm_data_parallel_info",
    "get_client_port_for_rank",
    "get_shard_info",
]
