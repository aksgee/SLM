# Executive Brief: Quantization Effect on performance and latency of sub-1B SLM Multi-Agent System : A viable framework to find the right fit

## What it is
This research is a **Multi-Seed Benchmark** designed to evaluate the effect of quantization on **Multi-Agent System (MAS)** across different levels of model quantization. Using the sub-1B SLM (`qwen3:0.6b`) model, the research tested the standard MAS architecture (a Router Agent delegating to Specialized Agents) in five different precision states: **FP32, FP16, 8-bit (Q8_0), 4-bit (Q4_K_M), and 4-bit (Q4_K_S)**. The evaluation was conducted over a robust dataset of 150 prompts across multiple seeds to ensure statistically significant results.

## What we want to demonstrate
The primary goal is to determine if the performance degradation is typically caused by model quantization in  multi-agent systems and whether multi-agent systems provides a robust. We want to prove that we can aggressively shrink the model's memory footprint (by quantization) in MAS to run on highly constrained edge devices (like laptop) without suffering a catastrophic loss in accuracy and at the same significantly reduce the latency.

## What was the result
The research demonstrated that the MAS architecture is remarkably resilient to quantization:
- **FP32 and FP16**: Established the baseline with high accuracies of 94.7% and 92.7%, respectively, but required significant VRAM (up to ~3.5 GB) and had higher latency.
- **The Sweet Spot (8-bit Q8_0)**: Maintained an excellent accuracy of **92.3%**, while cutting VRAM usage by nearly half (to **1810 MB**) and significantly improving generation speed (4622ms average time).
- **The Viable Edge (4-bit Q4_K_M)**: Surprisingly, this medium 4-bit quantization held a strong **90.3% accuracy** while dropping VRAM requirements to just **1500 MB** and latency to 3363ms.
- **The Breaking Point (4-bit Q4_K_S)**: The most aggressive 4-bit quantization finally showed a significant drop in overall accuracy (76.0%), though interestingly, the Router Agent remained highly accurate (95.3%).

## How can this research be applied
These results are highly actionable for edge AI deployments. This configuration can be applied to:
- **On-Device AI**: Running complex virtual assistants entirely locally on smartphones, tablets, or laptops without dedicated GPUs, where available RAM/VRAM is strictly limited to 2 - 8 GB.
- **High-Concurrency Servers**: Allowing cloud providers to pack significantly more agentic instances onto a single GPU by using the Q4_K_M or Q8_0 variants without a meaningful drop in reliability.

## What is the benefit
This research proves a critical hypothesis: **quantization degradation in MAS is not catastropic**. Unlike single large models that completely break down at 4-bit precision, a multi-agent system of 4-bit models remains highly capable. The core benefit is the ability to deploy enterprise-grade, highly accurate (>90%) agentic workflows on low-cost, low-power consumer hardware, unlocking massive savings in infrastructure costs and enabling true privacy-first local processing.

