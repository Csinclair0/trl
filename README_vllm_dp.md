# Data-Parallel vLLM Serving for Faster Generation

This feature implements data-parallel vLLM serving to address the performance bottlenecks in generation, especially for reasoning models as described in [GitHub issue #3195](https://github.com/huggingface/trl/issues/3195).

## Problem

In the open-r1 project on 7B Instruct and Reasoning models for code and mathematics datasets, generation time can become a significant bottleneck, sometimes taking over 5 minutes for a single generation. This can severely limit throughput during training and inference.

## Solution

Our solution implements a data-parallel approach for vLLM serving, which:

1. Distributes a batch of prompts across multiple vLLM server instances
2. Each instance processes its portion of the batch in parallel
3. Results are combined back into a complete response
4. Effectively increases throughput for larger batch sizes

## Components

The implementation consists of three main components:

1. **Data-Parallel Server (`vllm_serve_dp.py`)**: Spawns multiple vLLM server instances, each on a separate set of GPUs, distributing tensor parallelism within each instance.

2. **Data-Parallel Client (`vllm_dp_client.py`)**: Connects to all server instances, distributes prompts across them, and combines results back in the correct order.

3. **Benchmarking Tool (`benchmark_vllm_dp.py`)**: Evaluates the performance of data-parallel serving compared to standard vLLM serving.

## Usage

### Starting a Data-Parallel vLLM Server

To start a data-parallel vLLM server with 4 instances, each using 2 GPUs for tensor parallelism:

```bash
trl vllm-serve-dp \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tensor_parallel_size 2 \
    --data_parallel_size 4
```

This will start 4 server instances, each using 2 GPUs, occupying a total of 8 GPUs. Each instance will be available on a different port starting from the default 8000 (configurable).

### Using the Data-Parallel Client

To interact with the data-parallel server from Python:

```python
from trl.extras.vllm_dp_client import VLLMDataParallelClient

# Connect to 4 data-parallel instances at the default ports
client = VLLMDataParallelClient(dp_size=4)

# Generate completions
responses = client.generate([
    "Explain quantum physics",
    "Write a poem about stars",
    "Solve this math problem: 3x + 5 = 14",
    "Write a function to calculate Fibonacci numbers"
])

# Process the results
for prompt, completion in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Completion: {completion}")
    print()
```

### Benchmarking

To compare data-parallel serving with standard vLLM serving across different batch sizes:

```bash
trl benchmark-vllm-dp \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset open-r1/OpenR1-Math-cn_k12-86k \
    --num_prompts 128 \
    --batch_sizes 1,4,8,16,32,64,128 \
    --data_parallel_sizes 1,2,4 \
    --tensor_parallel_size 2 \
    --output_dir vllm_benchmark_results \
    --auto_launch true
```

This will:
- Load 128 prompts from the specified dataset
- Test different batch sizes: 1, 4, 8, 16, 32, 64, 128
- Test different data parallel sizes: 1 (standard), 2, 4
- Configure each instance to use 2 GPUs for tensor parallelism
- Automatically launch server instances for testing
- Save results and plots to the specified output directory

## Performance Considerations

- Data parallelism works best for large batch sizes
- There is an optimal batch size for each configuration, typically around 32 as found in our benchmarking
- For single-prompt generation, standard vLLM serving may still be more efficient
- When using multiple server instances, make sure to have enough GPU memory available

## Implementation Notes

- Each data-parallel instance gets assigned a specific set of GPUs
- Ports are assigned sequentially (default starts at 8000)
- The client distributes prompts evenly across instances, with any remainder going to the first few instances
- Results are reassembled in the original prompt order

## Requirements

- vLLM
- FastAPI
- Pydantic
- Uvicorn
- Requests

## Future Improvements

- Support for asynchronous, non-blocking generation
- Dynamic load balancing based on prompt length
- Integration with training loops for efficient gradient accumulation
- Improved error handling and retry mechanisms 