# KV cache nanoGPT Generation Benchmark

## Configuration

- Model: GPT-2 124M
- Device: Apple MPS
- Batch size: 1
- Generated tokens: 200
- Warmup runs: 3
- Measured runs: 5
- `torch.compile`: False
- `top_k`: 200
- `use_cache`: True

## Results

| Prompt Tokens | Generated Tokens | Median Latency (s) | Throughput (tokens/s) |
|---:|---:|---:|---:|
| 16  | 200 | 3.594 | 55.64 |
| 64  | 200 | 3.649 | 54.81 |
| 128 | 200 | 3.730 | 53.63 |
| 256 | 200 | 3.833 | 52.18 |
| 512 | 200 | 4.851 | 41.23 |