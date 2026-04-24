# Reducing Cognitive Load via Specialized Routing: A Multi‑Agent Tool Calling System with a sub-1B parameter language model

## Abstract
Tool calling is a core capability for enterprise LM agents, but small models (≤1B parameters) struggle with multi‑tool selection and structured output generation. A multi‑agent architecture is proposed that decomposes the task into a lightweight **Router Agent** (classifies the query into specific domains) and **Specialized Agents** (each handling exactly one tool type). Using a 0.6B Qwen3 model, this system achieves **96.0% accuracy** on a 25‑query benchmark, outperforming the same model in zero‑shot (64%) and few‑shot (76%) configurations. Compared to much larger models - 7B DeepSeek model (72% accuracy, 22s latency) and a 14B Phi‑4 model (96% accuracy, 8s latency), the multi‑agent system offers a compelling trade‑off: near‑state‑of‑the‑art accuracy at **4s average latency** and minimal memory footprint. The results demonstrate that **task decomposition and role specialization** can compensate for model size, enabling efficient and accurate tool calling on sub‑billion parameter models if given an efficient prompt template.

## 1. Introduction
Language models (LMs) are increasingly used as autonomous agents that invoke external tools. However, deploying even modestly sized models (7B parameters) on edge devices or in latency‑sensitive environments remains challenging. Sub‑billion parameter models (e.g., 0.6B) are attractive for their speed and low memory, but they often fail at multi‑tool calling because they must simultaneously: 
(i) understand the user intent, 
(ii) select the correct tool from a set, and 
(iii) generate well‑structured JSON with required parameters.

**Multi‑agent systems (MAS)** offer a solution: Instead of a single model that handles all tools, a team of specialized agents divides the work. In this paper, a two‑stage MAS for enterprise tool calling is designed and evaluated:
- A **Router Agent** that classifies the query into categories.
- **Specialized Agents**, each trained (via prompting) to generate a single tool call.

All agents use the same 0.6B Qwen3 model. The MAS is compared against single‑model baselines (Qwen3‑0.6B, DeepSeek‑R1‑7B, Phi‑4) in zero‑shot and few‑shot settings. The main contributions are:
1. A concrete multi‑agent architecture that reduces cognitive load by separating routing from tool execution.
2. Empirical evidence that a 0.6B MAS achieves 96% accuracy – competitive with a 14B model (96%) but at 3× lower latency and much smaller memory.

Most evaluations use large models (7B+) for this kind of study. The work presented here systematically compares a multi‑agent architecture against single‑model baselines for tool calling using a **0.6B parameter model**, showing that decomposition can bridge the accuracy gap with much larger models.

## 3. Methodology

### 3.1 Multi‑Agent Architecture
The system consists of five agents (all using `qwen3:0.6b` with temperature 0.1):
- **Router Agent** – Input: user query. Output: JSON `{"category": "...", "priority": "...", "reason": "..."}`. Categories: `IT_TICKET`, `LEAVE_REQUEST`, `CUSTOMER_QUERY`, `KNOWLEDGE_QUERY`. The prompt includes 10 examples covering all categories.
- **Specialized Aggent** - Responsible for a single domain. Input: Domain Specific Prompts, Few Shot Examples, Constraint.
  - **IT Ticket Agent** – Input: original query. Output: `{"name": "create_it_ticket", "parameters": {"title": "...", "description": "...", "priority": "Low|Medium|High|Critical", "department": "..."}}`.
  - **Leave Request Agent** – Output: `{"name": "process_leave_request", "parameters": {"employee_id": "...", "leave_type": "Sick|Casual|Earned", "start_date": "...", "end_date": "...", "reason": "..."}}`.
  - **Customer Query Agent** – Output: `{"name": "route_customer_query", "parameters": {"query_type": "...", "customer_id": "...", "priority": "Low|Medium|High", "department": "..."}}`.
  - **Knowledge Agent** – Output: `{"name": "get_internal_knowledge", "parameters": {"topic": "...", "department": "..."}}`.

Each specialized agent’s prompt contains only its own tool schema and a few examples. The orchestrator calls the router first, then passes the original query to the appropriate specialist. No iterative refinement is used; one‑pass accuracy is measured.

### 3.2 Baseline Models
Three single‑model configurations are used for comparison:
- **qwen3:0.6b** – the same base model, prompted with all four tools and asked to output a single JSON object (zero‑shot and few‑shot).
- **deepseek-r1:7b** – a 7B reasoning model (zero‑shot and few‑shot).
- **phi4:latest** – a 14B model (zero‑shot and few‑shot).

All baselines use the same tool schemas and output format. Few‑shot prompts include four examples (one per tool).

### 3.3 Dataset and Evaluation
A dataset of **25 natural language queries** balanced across the four categories is used. Each query has a ground‑truth tool name and expected parameter values. Evaluation criteria are defined in `evaluation_criteria.jsonl` and include:
- `expected_name`: the tool name.
- `required_keys`: mandatory parameter keys.


A response is considered **successful** if: 
(i) the tool name matches,and 
(ii) all required keys are present.   
Average latency (end‑to‑end, including router+specialist for MAS) is also measured.

### 3.4 Hardware and Software
All experiments are run on a MacBook Pro with M2 processor and 16 GB RAM, using Ollama v0.5.1. Latency is measured in milliseconds (ms).

## 4. Results

### 4.1 Overall Accuracy and Latency

| Model/Approach               | Accuracy | Avg Time (ms) | Success/Total |
|------------------------------|----------|---------------|---------------|
| qwen3:0.6b (zero‑shot)       | 64.0%    | 4034          | 16/25         |
| qwen3:0.6b (few‑shot)        | 76.0%    | 3563          | 19/25         |
| deepseek-r1:7b (zero‑shot)   | 68.0%    | 18160         | 13/25         |
| deepseek-r1:7b (few‑shot)    | 68.0%    | 17559         | 18/25         |
| phi4:latest (zero‑shot)      | 84.0%    | 7834          | 21/25         |
| phi4:latest (few‑shot)       | 96.0%    | 8171          | 24/25         |
| **Multi‑Agent (qwen3:0.6b)** | **96.0%**| **4253**      | **24/25**     |

<img width="1830" height="778" alt="image" src="https://github.com/user-attachments/assets/7afb0449-b346-4020-853a-3d597284ccd9" />

Key observations:
- The multi‑agent system improves the base 0.6B model by **32 percentage points** over zero‑shot and **20 points** over few‑shot.
- It matches the accuracy of phi4‑few‑shot (96%) and is 12 points improvement over phi4‑zero‑shot (84%), while being **2x faster** (4253 ms vs. 8171 ms).
- The 7B DeepSeek model performs poorly (68%) and is an order of magnitude slower (~18 s), indicating that larger size alone does not guarantee better tool calling; prompt sensitivity and architecture matter. This is explainable considering DeepSeek is a reasoning model.  

### 4.2 Accuracy by Category (Multi‑Agent)

| Category        | Count | Accuracy |
|-----------------|-------|----------|
| IT_TICKET       | 6     | 100.0%   |
| LEAVE_REQUEST   | 6     | 100.0%   |
| CUSTOMER_QUERY  | 7     | 85.7%    |
| KNOWLEDGE_QUERY | 6     | 100.0%   |

<img width="1270" height="748" alt="image" src="https://github.com/user-attachments/assets/32d6dd98-a3e6-4a73-9b21-ab8c01416eba" />


The one error occurred in:
- **Customer Query** – a query about "Projector in boardroom is not working for today's client presentation" was routed to `CUSTOMER_QUERY` instead of `CREATE_IT_TICKET`, leading to a `wrong_tool` error.


All errors are **router errors**; once the correct category is chosen, the specialized agent never fails. This highlights that the bottleneck is intent classification, not parameter extraction.

### 4.3 Error Analysis

Across all models, the most common error is `wrong_tool` (misclassification) followed by parse error (malformed JSON output). For the single‑model baselines, the model sometimes confuses similar tools (e.g., `create_it_ticket` vs. `route_customer_query`). For the multi‑agent system, the one error is also routing error. No errors are due to missing required keys or invalid enum values, indicating that specialized agents are highly reliable.

### 4.4 Latency Comparison

- **Multi‑agent**: 4253 ms (router ~1000 ms, specialist ~3200 ms).
- **phi4 (zero‑shot)**: 8171 ms – ~2× slower.
- **deepseek (zero‑shot)**: 18160 ms – ~4× slower.

The multi‑agent system’s low latency makes it suitable for real‑time applications, whereas the 7B and 14B models introduce noticeable delays.

## 5. Discussion

### 5.1 Why Does the Multi‑Agent Approach Work?
Small LLMs have limited working memory and attention. A single prompt that describes four tools forces the model to keep all schemas in its context simultaneously, leading to confusion. By **splitting the task**, the per‑agent cognitive load is reduced:
- The router only distinguishes four categories – a simple classification task.
- Each specialist only remembers **one** tool schema and a few examples.

Furthermore, the specialist prompts contain domain‑specific heuristics (e.g., “Critical for production failures”) that would be diluted if all tools were mixed. This explains why the 0.6B MAS outperforms the same model in few‑shot (76% → 96%).

### 5.2 Comparison with Larger Models
Phi‑4 (14B) achieves 84% accuracy in zero‑shot but is slower and much larger (≈8 GB quantized vs. 0.5 GB for qwen3:0.6b). For edge deployments or high‑throughput APIs, the multi‑agent system offers a superior trade‑off: 96% accuracy at 4s latency with minimal memory. DeepSeek‑R1‑7B, despite being larger, performs worse – likely because it is a reasoning model not optimized for strict JSON output and tool selection.

### 5.3 Limitations
- The dataset is small (25 queries). A larger, more diverse benchmark is needed to confirm generalizability.
- The router remains the weak point; future work could fine‑tune a small classifier or use an ensemble of routers.
- The system does not handle multi‑tool requests (e.g., “create a ticket and send an email”). Extending the architecture to support sequences of tool calls is an obvious next step.

## 6. Conclusion
This work has shown that a multi‑agent system with a 0.6B parameter LLM can achieve **92% tool‑call accuracy** on an enterprise benchmark, outperforming the same model in single‑agent mode by 16 percentage points and matching the accuracy of a 14B model at 3× lower latency. The key idea – **decoupling intent classification from tool execution** – enables extremely small models to excel at structured tool calling. These results provide a practical recipe for building efficient, accurate, and low‑memory tool‑calling agents for edge and real‑time applications.

## References
[1] T. Schick et al. “Toolformer: Language models can teach themselves to use tools.” *arXiv:2302.04761*, 2023.  
[2] S. G. Patil et al. “Gorilla: Large language models connected with massive APIs.” *arXiv:2305.15334*, 2023.  
[3] Z. Liu et al. “Small LLMs can be efficient tool callers with structured prompting.” *EMNLP 2024 Industry Track*.  
[4] S. Hong et al. “MetaGPT: Meta programming for multi‑agent collaborative software engineering.” *arXiv:2308.00352*, 2023.  
[5] Q. Wu et al. “AutoGen: Enabling next‑gen LLM applications via multi‑agent conversation.” *arXiv:2308.08155*, 2023.

**Appendix** – Detailed logs and evaluation criteria are available in the result folder (`detailed_toolcalling_log.txt`, `multi_agent_log.txt`, etc.).
