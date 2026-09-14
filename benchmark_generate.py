import time
import torch
import tiktoken
import math
import statistics

from model import GPT

device = "mps"

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)
max_new_tokens = 200

enc = tiktoken.get_encoding("gpt2")

base_tokens = enc.encode(
    "The future of artificial intelligence is going to change "
    "the way people interact with computers. "
)

prompt_lengths = [16, 64, 128, 256, 512]


for length in prompt_lengths:
    repeats = math.ceil(length / len(base_tokens))
    tokens = (base_tokens * repeats)[:length]

    idx = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    )[None, :]

    # warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=200,
                use_cache=True
            )

    torch.mps.synchronize()

    # benchmark
    times = []
    for _ in range(5):
        torch.mps.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=200,
                use_cache=True
            )

        torch.mps.synchronize()

        times.append(time.perf_counter() - start)

    # Metrics
    median_latency = statistics.median(times)
    throughput = max_new_tokens / median_latency

    print(f"Prompt length: {length} tokens")
    print(f"Generated:     {max_new_tokens} tokens")
    print(f"Latency:       {median_latency:.3f} sec")
    print(f"Throughput:    {throughput:.2f} tokens/sec")
    print("-" * 40)