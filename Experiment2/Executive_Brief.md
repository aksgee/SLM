# Executive Brief: Experiment 16 (Quantized Multi-Agent System Comparison)

## What it is
Experiment 16 is a **Multi-Seed Benchmark** designed to evaluate the performance of a **Multi-Agent System (MAS)** across different levels of model quantization. Using the `qwen3:0.6b` model, the experiment tested the standard MAS architecture (a Router Agent delegating to Specialized Agents) in five different precision states: **FP32, FP16, 8-bit (Q8_0), 4-bit (Q4_K_M), and 4-bit (Q4_K_S)**. The evaluation was conducted over a robust dataset of 301 prompts across multiple seeds to ensure statistically significant results.

## What we want to demonstrate
The primary goal is to determine if the architectural benefits of a Multi-Agent System—specifically reduced cognitive load and specialized prompting—can compensate for the performance degradation typically caused by model quantization. We want to prove that we can aggressively shrink the model's memory footprint to run on highly constrained edge devices without suffering a catastrophic loss in tool-calling accuracy.

## What was the result
The experiment demonstrated that the MAS architecture is remarkably resilient to quantization, with deeper insights revealed by the new error tracking and specialized agent metrics:
- **FP32 and FP16**: Established the baseline with high overall accuracies of 94.7% and 92.7%. The Specialized Agents were highly effective (98.6% and 97.5% accuracy given correct routing), and the few errors were primarily 'Wrong Tool' selections or minor enum mismatches.
- **The Sweet Spot (8-bit Q8_0)**: Maintained an excellent overall accuracy of **92.3%** and an impressive Specialized Agent accuracy of **98.2%**. It cut VRAM usage by nearly half (to **1810 MB**) and significantly improved generation speed (4622ms average time) while exhibiting zero JSON parsing errors.
- **The Viable Edge (4-bit Q4_K_M)**: Surprisingly, this medium 4-bit quantization held a strong **90.3% overall accuracy** (96.1% specialized agent accuracy). VRAM requirements dropped to just **1500 MB**, and errors remained low, with only slight increases in missing keys or invalid enum errors.
- **The Breaking Point (4-bit Q4_K_S)**: The most aggressive 4-bit quantization showed a significant drop in overall accuracy (76.0%). Interestingly, the Router Agent remained highly accurate (95.3%), but the Specialized Agents degraded sharply (79.7% accuracy). The detailed error logs reveal this was driven by a massive spike in 'Missing Required Key' (36 instances) and 'JSON Errors' (16 instances), proving the model lost its ability to reliably generate structured schemas at this compression level.

## How can this experiment be applied
These results are highly actionable for edge AI deployments. This configuration can be applied to:
- **On-Device AI**: Running complex virtual assistants entirely locally on smartphones, tablets, or laptops without dedicated GPUs, where available RAM/VRAM is strictly limited to 1.5 - 2 GB.
- **High-Concurrency Servers**: Allowing cloud providers to pack significantly more agentic instances onto a single GPU by using the Q4_K_M or Q8_0 variants without a meaningful drop in reliability.

## What is the benefit
This experiment proves a critical hypothesis: **Specialization protects against quantization degradation**. Unlike single large models that completely break down at 4-bit precision (as seen in previous experiments), a multi-agent system of 4-bit models remains highly capable. The core benefit is the ability to deploy enterprise-grade, highly accurate (>90%) agentic workflows on low-cost, low-power consumer hardware, unlocking massive savings in infrastructure costs and enabling true privacy-first local processing.
