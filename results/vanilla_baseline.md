# Vanilla nanoGPT Generation Benchmark

## Configuration

- Model: GPT-2 124M
- Device: Apple MPS
- Batch size: 1
- Generated tokens: 200
- Warmup runs: 3
- Measured runs: 5
- `torch.compile`: False
- `top_k`: 200

## Results

| Prompt Tokens | Generated Tokens | Median Latency (s) | Throughput (tokens/s) |
|---:|---:|---:|---:|
| 16  | 200 | 7.741  | 25.84 |
| 64  | 200 | 8.489  | 23.56 |
| 128 | 200 | 10.568 | 18.93 |
| 256 | 200 | 14.602 | 13.70 |
| 512 | 200 | 23.164 | 8.63  |