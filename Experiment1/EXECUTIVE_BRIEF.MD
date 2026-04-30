# Executive Brief: Multi-Agent System vs. Single Model Tool Calling -  A hypothesis to prove that specialization reduces hallucination and improves accuracy.

## What it is
This research aims to evaluate the performance of a **Multi-Agent System** architecture against standard **Single Large Language Model (SLM)** approaches for enterprise tool-calling tasks. Instead of relying on a single model to handle a large, complex schema of multiple tools, the Multi-Agent architecture introduces a **Router Agent** that classifies incoming queries and delegates them to one of four **Specialized Agents** (IT Ticket, Leave Request, Customer Query, or Knowledge Query). All agents in the multi-agent setup utilize a highly efficient, small-parameter model (`qwen3:0.6b`).

## What we want to demonstrate
The primary goal of this experiment is to demonstrate that a decentralized, multi-agent architecture using very small models (0.6B parameters) can match or outperform larger, more resource-intensive single models (such as `deepseek-r1:7b` or `phi4:latest`). We hypothesize that by reducing the cognitive load, providing domain-specific prompts, and narrowing the search space for each specialized agent, we can achieve higher accuracy, lower latency, and fewer tool-calling errors.

## What was the result
The experiment yielded highly successful results:
- **High Accuracy**: The Multi-Agent System (`qwen3:0.6b`) achieved a **96.0% accuracy** rate, matching the best-performing large model (`phi4:latest` with few-shot prompting) and significantly outperforming a single `qwen3:0.6b` model (76.0% accuracy). 
- **Low Latency**: The Multi-Agent System achieved this high accuracy with an average latency of **4253ms**, which is nearly twice as fast as `phi4:latest` (8171ms) and significantly faster than `deepseek-r1:7b` (~17500ms).
- **Reduced Errors**: The multi-agent approach almost entirely eliminated parsing errors and missing parameter errors, with only a single routing error observed across all test cases.

## How can this experiment be applied
This architecture is directly applicable to complex enterprise automation scenarios, including:
- **Enterprise Virtual Assistants**: Handling employee requests across IT, HR, and Finance seamlessly.
- **Customer Support Automation**: Intelligently routing user queries to the correct department and extracting necessary tool parameters.
- **Helpdesk & Ticketing Systems**: Automatically structuring unstructured requests into actionable tickets or database queries.
- **Internal Knowledge Retrieval**: Directing specific policy or procedural questions to the right knowledge bases.

## What is the benefit
Implementing this Multi-Agent architecture provides several strategic benefits:
1. **Cost and Compute Efficiency**: By utilizing highly capable 0.6B parameter models instead of 7B-14B parameter models, enterprises can drastically reduce hosting costs, API usage, and VRAM requirements.
2. **Superior Speed**: Faster response times improve user experience, allowing real-time or near-real-time agentic workflows.
3. **High Reliability**: Specialization reduces hallucinations and improves accuracy. Specialized agents are less prone to hallucination, as they only need to understand one specific tool rather than a massive schema.
4. **Maintainability**: New tools or capabilities can be added by simply creating a new specialized agent and updating the router, without the need to continuously re-prompt or fine-tune a massive central model.
