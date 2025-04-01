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
import csv
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from trl import TrlParser
from trl.extras.vllm_client import VLLMClient
from trl.extras.vllm_dp_client import VLLMDataParallelClient
from trl.import_utils import is_datasets_available


if is_datasets_available():
    from datasets import load_dataset, Dataset


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the benchmark script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        dataset (`str`):
            Dataset to use for prompts. Defaults to open-r1/OpenR1-Math-cn_k12-86k.
        dataset_split (`str`):
            Dataset split to use.
        num_prompts (`int`):
            Number of prompts to use for benchmarking. If larger than dataset, will repeat prompts.
        batch_sizes (`str`):
            Comma-separated list of batch sizes to benchmark.
        max_tokens (`int`):
            Maximum number of tokens to generate for each prompt.
        data_parallel_sizes (`str`):
            Comma-separated list of data parallel sizes to benchmark.
        tensor_parallel_size (`int`):
            Number of GPUs to use for tensor parallelism.
        output_dir (`str`):
            Directory to save benchmark results.
        auto_launch (`bool`):
            Whether to automatically launch vLLM servers.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    dataset: str = field(
        default="open-r1/OpenR1-Math-cn_k12-86k",
        metadata={"help": "Dataset to use for prompts."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use."},
    )
    num_prompts: int = field(
        default=128,
        metadata={"help": "Number of prompts to use for benchmarking."},
    )
    batch_sizes: str = field(
        default="1,4,8,16,32,64,128",
        metadata={"help": "Comma-separated list of batch sizes to benchmark."},
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of tokens to generate for each prompt."},
    )
    data_parallel_sizes: str = field(
        default="1,2,4,8",
        metadata={"help": "Comma-separated list of data parallel sizes to benchmark."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs to use for tensor parallelism."},
    )
    output_dir: str = field(
        default="vllm_benchmark_results",
        metadata={"help": "Directory to save benchmark results."},
    )
    auto_launch: bool = field(
        default=False,
        metadata={"help": "Whether to automatically launch vLLM servers."},
    )


def load_prompts(dataset_name: str, split: str, num_prompts: int) -> List[str]:
    """
    Load prompts from a dataset.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Split of the dataset to use.
        num_prompts: Number of prompts to return.
        
    Returns:
        List of prompts.
    """
    if not is_datasets_available():
        raise ImportError("datasets is not installed. Please install it with `pip install datasets`.")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Check if dataset has 'problem' or 'prompt' field
    if "problem" in dataset.column_names:
        text_field = "problem"
    elif "prompt" in dataset.column_names:
        text_field = "prompt"
    else:
        # Try to find a suitable text field
        text_fields = [field for field in dataset.column_names if "text" in field.lower()]
        if text_fields:
            text_field = text_fields[0]
            logger.warning(f"Guessing text field as {text_field}")
        else:
            raise ValueError(
                f"Could not determine text field in dataset. Available columns: {dataset.column_names}"
            )
    
    # Get prompts
    prompts = [item[text_field] for item in dataset]
    
    # Handle case where we need more prompts than dataset size
    if num_prompts > len(prompts):
        # Repeat prompts to reach desired count
        factor = (num_prompts // len(prompts)) + 1
        prompts = prompts * factor
    
    # Return exact number of prompts
    return prompts[:num_prompts]


def launch_vllm_server(
    model: str, 
    tensor_parallel_size: int = 1, 
    data_parallel_size: int = 1, 
    base_port: int = 8000,
) -> Dict[str, Any]:
    """
    Launch vLLM server(s) with the specified configuration.
    
    Args:
        model: Model to serve.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        data_parallel_size: Number of data parallel instances to launch.
        base_port: Base port for the first server.
        
    Returns:
        Dictionary with process information.
    """
    if data_parallel_size > 1:
        # Launch data-parallel vLLM server
        cmd = [
            "python", "-m", "trl.scripts.vllm_serve_dp",
            "--model", model,
            "--tensor_parallel_size", str(tensor_parallel_size),
            "--data_parallel_size", str(data_parallel_size),
            "--port", str(base_port),
        ]
    else:
        # Launch standard vLLM server
        cmd = [
            "python", "-m", "trl.scripts.vllm_serve",
            "--model", model,
            "--tensor_parallel_size", str(tensor_parallel_size),
            "--port", str(base_port),
        ]
    
    # Start the process
    logger.info(f"Launching vLLM server with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    # Give server time to start up
    time.sleep(30)
    
    return {
        "process": process,
        "cmd": cmd,
        "data_parallel_size": data_parallel_size,
        "tensor_parallel_size": tensor_parallel_size,
    }


def stop_vllm_server(server_info: Dict[str, Any]) -> None:
    """
    Stop the vLLM server process.
    
    Args:
        server_info: Server information returned by launch_vllm_server.
    """
    process = server_info["process"]
    if process.poll() is None:  # Still running
        logger.info("Stopping vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not terminate gracefully, killing...")
            process.kill()


def benchmark_standard_server(
    prompts: List[str],
    batch_sizes: List[int],
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Benchmark a standard vLLM server (non-data-parallel).
    
    Args:
        prompts: List of prompts to use.
        batch_sizes: List of batch sizes to benchmark.
        max_tokens: Maximum number of tokens to generate per prompt.
        
    Returns:
        Dictionary with benchmark results.
    """
    client = VLLMClient()
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking with batch_size={batch_size}")
        
        # Run benchmark for this batch size
        times = []
        
        # Process prompts in chunks of batch_size
        for start_idx in range(0, len(prompts), batch_size):
            end_idx = min(start_idx + batch_size, len(prompts))
            batch = prompts[start_idx:end_idx]
            
            # Measure generation time
            start_time = time.time()
            client.generate(batch, max_tokens=max_tokens)
            end_time = time.time()
            
            # Record time
            elapsed = end_time - start_time
            times.append(elapsed)
            logger.info(f"  Batch {start_idx // batch_size + 1}: {elapsed:.2f} seconds")
        
        # Calculate statistics
        results[str(batch_size)] = {
            "times": times,
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "throughput": len(prompts) / sum(times),  # prompts per second
        }
        
        logger.info(f"  Average time: {results[str(batch_size)]['mean_time']:.2f} seconds")
        logger.info(f"  Throughput: {results[str(batch_size)]['throughput']:.2f} prompts/second")
    
    return results


def benchmark_dp_server(
    prompts: List[str],
    batch_sizes: List[int],
    data_parallel_size: int,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Benchmark a data-parallel vLLM server.
    
    Args:
        prompts: List of prompts to use.
        batch_sizes: List of batch sizes to benchmark.
        data_parallel_size: Number of data parallel instances.
        max_tokens: Maximum number of tokens to generate per prompt.
        
    Returns:
        Dictionary with benchmark results.
    """
    client = VLLMDataParallelClient(dp_size=data_parallel_size)
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking DP={data_parallel_size} with batch_size={batch_size}")
        
        # Run benchmark for this batch size
        times = []
        
        # Process prompts in chunks of batch_size
        for start_idx in range(0, len(prompts), batch_size):
            end_idx = min(start_idx + batch_size, len(prompts))
            batch = prompts[start_idx:end_idx]
            
            # Measure generation time
            start_time = time.time()
            client.generate(batch, max_tokens=max_tokens)
            end_time = time.time()
            
            # Record time
            elapsed = end_time - start_time
            times.append(elapsed)
            logger.info(f"  Batch {start_idx // batch_size + 1}: {elapsed:.2f} seconds")
        
        # Calculate statistics
        results[str(batch_size)] = {
            "times": times,
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "throughput": len(prompts) / sum(times),  # prompts per second
        }
        
        logger.info(f"  Average time: {results[str(batch_size)]['mean_time']:.2f} seconds")
        logger.info(f"  Throughput: {results[str(batch_size)]['throughput']:.2f} prompts/second")
    
    return results


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    model: str,
    tensor_parallel_size: int,
) -> None:
    """
    Save benchmark results to disk.
    
    Args:
        results: Dictionary of benchmark results.
        output_dir: Directory to save results.
        model: Model name.
        tensor_parallel_size: Tensor parallel size used.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw results as JSON
    model_name = model.split("/")[-1]
    results_file = output_path / f"results_{model_name}_tp{tensor_parallel_size}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary as CSV
    summary_file = output_path / f"summary_{model_name}_tp{tensor_parallel_size}.csv"
    
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["batch_size", "dp_size", "mean_time", "median_time", "min_time", "max_time", "throughput"]
        writer.writerow(header)
        
        # Write data
        for dp_size, dp_results in results.items():
            for batch_size, batch_results in dp_results.items():
                row = [
                    batch_size,
                    dp_size,
                    batch_results["mean_time"],
                    batch_results["median_time"],
                    batch_results["min_time"],
                    batch_results["max_time"],
                    batch_results["throughput"],
                ]
                writer.writerow(row)
    
    logger.info(f"Results saved to {output_path}")


def plot_results(
    results: Dict[str, Any],
    output_dir: str,
    model: str,
    tensor_parallel_size: int,
) -> None:
    """
    Create plots of benchmark results.
    
    Args:
        results: Dictionary of benchmark results.
        output_dir: Directory to save plots.
        model: Model name.
        tensor_parallel_size: Tensor parallel size used.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract model name
    model_name = model.split("/")[-1]
    
    # Prepare data for plotting
    batch_sizes = []
    for dp_results in results.values():
        batch_sizes.extend(list(dp_results.keys()))
    batch_sizes = sorted(list(set(batch_sizes)), key=int)
    
    dp_sizes = sorted(list(results.keys()), key=int)
    
    # Plot mean generation time vs batch size for each DP size
    plt.figure(figsize=(12, 8))
    
    for dp_size in dp_sizes:
        times = []
        for batch_size in batch_sizes:
            if batch_size in results[dp_size]:
                times.append(results[dp_size][batch_size]["mean_time"])
            else:
                times.append(np.nan)
        
        plt.plot(batch_sizes, times, marker='o', linewidth=2, label=f"DP={dp_size}")
    
    plt.xlabel("Batch Size", fontsize=14)
    plt.ylabel("Mean Generation Time (s)", fontsize=14)
    plt.title(f"Generation Time vs Batch Size - {model_name} (TP={tensor_parallel_size})", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    time_plot_file = output_path / f"time_vs_batch_{model_name}_tp{tensor_parallel_size}.png"
    plt.savefig(time_plot_file, dpi=300)
    
    # Plot throughput vs batch size for each DP size
    plt.figure(figsize=(12, 8))
    
    for dp_size in dp_sizes:
        throughputs = []
        for batch_size in batch_sizes:
            if batch_size in results[dp_size]:
                throughputs.append(results[dp_size][batch_size]["throughput"])
            else:
                throughputs.append(np.nan)
        
        plt.plot(batch_sizes, throughputs, marker='o', linewidth=2, label=f"DP={dp_size}")
    
    plt.xlabel("Batch Size", fontsize=14)
    plt.ylabel("Throughput (prompts/s)", fontsize=14)
    plt.title(f"Throughput vs Batch Size - {model_name} (TP={tensor_parallel_size})", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    throughput_plot_file = output_path / f"throughput_vs_batch_{model_name}_tp{tensor_parallel_size}.png"
    plt.savefig(throughput_plot_file, dpi=300)
    
    logger.info(f"Plots saved to {output_path}")


def main(script_args: ScriptArguments):
    # Parse batch sizes and DP sizes
    batch_sizes = [int(size) for size in script_args.batch_sizes.split(",")]
    dp_sizes = [int(size) for size in script_args.data_parallel_sizes.split(",")]
    
    # Load prompts
    logger.info(f"Loading prompts from {script_args.dataset}")
    prompts = load_prompts(script_args.dataset, script_args.dataset_split, script_args.num_prompts)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Store results
    all_results = {}
    server_process = None
    
    try:
        # Run benchmarks for each DP size
        for dp_size in dp_sizes:
            if script_args.auto_launch:
                # Launch server with appropriate configuration
                server_info = launch_vllm_server(
                    script_args.model,
                    script_args.tensor_parallel_size,
                    dp_size,
                )
                server_process = server_info["process"]
            
            # Wait for server to start
            time.sleep(10)
            
            # Run appropriate benchmark
            if dp_size == 1:
                # Standard server
                all_results[str(dp_size)] = benchmark_standard_server(
                    prompts,
                    batch_sizes,
                    script_args.max_tokens,
                )
            else:
                # Data-parallel server
                all_results[str(dp_size)] = benchmark_dp_server(
                    prompts,
                    batch_sizes,
                    dp_size,
                    script_args.max_tokens,
                )
            
            # Stop server if we launched it
            if script_args.auto_launch and server_process is not None:
                stop_vllm_server(server_info)
                server_process = None
    
    finally:
        # Ensure server is stopped if we launched it
        if script_args.auto_launch and server_process is not None:
            stop_vllm_server({"process": server_process})
    
    # Save and plot results
    save_results(
        all_results,
        script_args.output_dir,
        script_args.model,
        script_args.tensor_parallel_size,
    )
    
    plot_results(
        all_results,
        script_args.output_dir,
        script_args.model,
        script_args.tensor_parallel_size,
    )


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "benchmark-vllm-dp", 
            help="Benchmark data-parallel vLLM serving",
            dataclass_types=ScriptArguments
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args) 