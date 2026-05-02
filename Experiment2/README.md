# Finding the right fit: Study of quantization effect on sub-1B MAS architecture

## Abstract
Deploying enterprise language model agents on edge devices requires aggressive optimization, often leading to performance degradation in small models. While Multi-Agent Systems (MAS) can overcome the cognitive limitations of sub-billion parameter models (e.g., 0.6B), the impact of quantization on such specialized architectures remains underexplored. This study evaluates a two-stage MAS (Router Agent and Specialized Agents) using the Qwen3-0.6B model across five quantization levels: FP32, FP16, 8-bit (Q8_0), and 4-bit (Q4_K_M, Q4_K_S). The results reveal that specialization provides remarkable resilience against quantization degradation. The 8-bit (Q8_0) configuration emerges as the optimal sweet spot, maintaining **92.3% accuracy** while reducing VRAM usage by nearly 50% (to 1.8 GB) and halving latency (4.6s) compared to the FP32 baseline (94.7% accuracy, 3.5 GB VRAM). Furthermore, the study identifies a structural breaking point at extreme 4-bit quantization (Q4_K_S), where accuracy drops to 76.0% due to catastrophic JSON parsing and schema generation failures, despite the routing logic remaining intact. These findings provide a practical framework for deploying highly accurate, low-memory MAS on resource-constrained edge devices.

## 1. Introduction
The demand for autonomous, tool-calling language models (LMs) is expanding from cloud infrastructure to edge devices, such as smartphones, laptops, and IoT hardware. This shift necessitates the use of sub-billion parameter models (e.g., 0.6B) to meet strict VRAM and latency constraints. However, as demonstrated in previous work, small models struggle to handle complex, multi-tool schemas in a single pass due to cognitive overload. 

Decomposing the task via a **Multi-Agent System (MAS)**—separating intent classification (Router) from task execution (Specialized Agents)—has proven highly effective, allowing a 0.6B model to match the tool-calling accuracy of a 14B model. Yet, to truly unlock edge deployments, these sub-1B models must be further compressed using quantization techniques. 

A critical open question remains: *Does the architectural advantage of MAS survive aggressive quantization, or do small models lose their reasoning and formatting capabilities when their precision is reduced?*

This paper presents a systematic multi-seed benchmark evaluating the Qwen3-0.6B MAS architecture across five precision states (FP32, FP16, Q8_0, Q4_K_M, and Q4_K_S). The main contributions are:
1. Empirical evidence that task specialization protects against the typical accuracy degradation associated with 8-bit and medium 4-bit quantization.
2. Identification of the "sweet spot" (Q8_0) for edge deployment, offering >92% accuracy under 2GB VRAM.
3. Detailed error analysis revealing that extreme 4-bit quantization (Q4_K_S) breaks the model's ability to adhere to structural formatting (JSON/schema), rather than its logical routing capabilities.

## 2. Methodology

### 2.1 Architecture Principle
The system relies on a decoupled, two-stage Multi-Agent Architecture:
- **Router Agent:** Analyzes the raw user query and outputs a lightweight JSON object classifying the intent into one of four categories (`IT_TICKET`, `LEAVE_REQUEST`, `CUSTOMER_QUERY`, `KNOWLEDGE_QUERY`).
- **Specialized Agents:** Four dedicated agents, each containing only the prompt instructions and tool schema relevant to its specific domain. The Orchestrator forwards the query to the correct specialist based on the Router's classification.

All agents are instantiated using the same underlying Qwen3-0.6B model.

### 2.2 Quantization Baselines
To measure the trade-offs of model compression, the MAS was evaluated using the following GGUF quantization formats:
- **FP32 (F32):** 32-bit floating point (Baseline uncompressed, ~3.5 GB VRAM)
- **FP16 (F16):** 16-bit floating point (~2.5 GB VRAM)
- **8-bit (Q8_0):** 8-bit integer quantization (~1.8 GB VRAM)
- **4-bit (Q4_K_M):** Medium 4-bit quantization, balancing compression and quality (~1.5 GB VRAM)
- **4-bit (Q4_K_S):** Small 4-bit quantization, prioritizing maximum compression (~1.4 GB VRAM)

### 2.3 Dataset and Evaluation
The evaluation utilized a robust dataset of **150 natural language enterprise queries**, spanning all four domains. To ensure statistical reliability and account for generative variance, the entire dataset was executed across two distinct random seeds (Seed 42 and Seed 999), resulting in **300 total evaluations** per model variant.

A response was marked **successful** only if:
1. The tool name perfectly matched the expected tool.
2. All required keys for that specific tool schema were present and populated with valid data (including strict enum validation where applicable).
3. The JSON was structurally sound and parsable.

### 2.4 Hardware
All experiments were conducted locally on consumer hardware (MacBook Pro, Apple Silicon) using Ollama to measure real-world inference latency and peak VRAM consumption.

## 3. Results

### 3.1 Overall Accuracy, Latency, and Memory Trade-offs

| Model / Precision | Accuracy (mean±std) | Router Acc | Specialized Agent Acc | Avg Time (ms) | Peak VRAM (MB) |
|-------------------|---------------------|------------|-----------------------|---------------|----------------|
| **FP32 (F32)**    | 94.7% ± 0.7%        | 96.0%      | 98.6%                 | 8996          | 3496           |
| **FP16 (F16)**    | 92.7% ± 0.7%        | 95.0%      | 97.5%                 | 6354          | 2487           |
| **8-bit (Q8_0)**  | 92.3% ± 0.3%        | 94.0%      | 98.2%                 | 4622          | 1810           |
| **4-bit (Q4_K_M)**| 90.3% ± 1.7%        | 94.0%      | 96.1%                 | 3363          | 1500           |
| **4-bit (Q4_K_S)**| 76.0% ± 1.3%        | 95.3%      | 79.7%                 | 3227          | 1449           |

**Key Observations:**
- **The Sweet Spot (Q8_0):** The 8-bit model maintained a highly competitive 92.3% overall accuracy (only a 2.4% drop from FP32) while reducing VRAM usage by 48% and nearly doubling generation speed. 
- **The Viable Edge (Q4_K_M):** The medium 4-bit variant successfully held above the 90% accuracy threshold, fitting the entire multi-agent workflow into just 1.5 GB of VRAM.
- **The Breaking Point (Q4_K_S):** The most aggressively quantized model suffered a severe 14.3% accuracy drop compared to its medium counterpart, indicating a catastrophic failure threshold.

### 3.2 Error Type Analysis

To understand *why* the models failed at higher quantization levels, errors across the 300 runs were categorized:

| Model / Precision | JSON Error | Missing Required Key | Invalid Enum/Value | Wrong Tool | Other | Total Errors |
|-------------------|------------|----------------------|--------------------|------------|-------|--------------|
| **FP32 (F32)**    | 1          | 0                    | 3                  | 12         | 0     | 16           |
| **FP16 (F16)**    | 1          | 2                    | 4                  | 15         | 0     | 22           |
| **8-bit (Q8_0)**  | 0          | 1                    | 4                  | 18         | 0     | 23           |
| **4-bit (Q4_K_M)**| 1          | 5                    | 6                  | 17         | 0     | 29           |
| **4-bit (Q4_K_S)**| 16         | 36                   | 7                  | 13         | 0     | 72           |

**Insight on Structural Collapse:**
While FP32, FP16, and Q8_0 errors were almost exclusively driven by semantic misunderstandings ("Wrong Tool" selection by the router), the Q4_K_S model's failure was entirely structural. It generated **36 missing key errors** and **16 raw JSON parsing errors**. Interestingly, the Q4_K_S Router Agent maintained a 95.3% accuracy rate; the breakdown occurred when the Specialized Agents attempted to generate the final, complex JSON schemas.

## 4. Discussion

### 4.1 Specialization Protects Against Quantization
Previous single-model experiments demonstrated that 4-bit quantization effectively destroys a 0.6B model's ability to perform tool calling. However, within a MAS architecture, the Q4_K_M model retained 90.3% accuracy. By isolating the cognitive load—allowing the model to focus purely on one schema and a few specialized examples—the architecture compensates for the loss of parameter precision. This proves that architectural design can act as a buffer against quantization degradation.

### 4.2 The Reality of Edge Deployments
For mobile and edge environments, VRAM is the primary constraint. The data strongly suggests that **8-bit (Q8_0)** is the optimal deployment configuration. It fits well within a 2 GB memory footprint, offers fast 4.6-second end-to-end latency (for a two-stage agent process), and virtually eliminates structural formatting errors. 

If extreme memory constraints exist (< 1.5 GB), **Q4_K_M** remains a viable fallback. However, practitioners must avoid pushing 0.6B models down to the **Q4_K_S** tier, as the precision loss critically impairs the model's ability to maintain JSON syntax and adhere to required parameter schemas.

## 5. Conclusion
This study investigated the impact of model quantization on a Multi-Agent System powered by a sub-1B parameter language model. The empirical results demonstrate that task decomposition not only boosts baseline accuracy but also insulates the model against compression penalties. The system reliably maintained >90% accuracy down to medium 4-bit precision, cutting VRAM requirements by over 50% compared to the FP32 baseline. By leveraging an 8-bit or medium 4-bit quantization strategy, organizations can deploy robust, enterprise-grade agentic workflows entirely locally on low-power, memory-constrained edge devices, achieving high reliability without the cost or latency of cloud-based inferences.
