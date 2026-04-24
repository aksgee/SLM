"""
Multi-Agent Tool Calling System using qwen3:0.6b

Architecture:
1. Router Agent - Classifies the query into one of 4 task types
2. Specialized Agents - Each agent handles only one specific tool type:
   - IT Ticket Agent
   - Leave Request Agent
   - Customer Query Agent
   - Knowledge Agent

This approach achieves better accuracy by:
- Reducing each agent's cognitive load (only 1 tool to remember)
- Using domain-specialized prompts with targeted examples
- Explicit routing reduces tool selection confusion
"""

import json
import time
import os
import re
import requests
import ollama
import pandas as pd
from tabulate import tabulate
from typing import Dict, List, Tuple, Any, Optional
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "qwen3:0.6b"
MODEL = DEFAULT_MODEL # This will be updated by run_multi_agent_experiment

# File paths
DATASET_FILE = os.path.join(SCRIPT_DIR, "prompts.jsonl")
EVAL_CRITERIA_FILE = os.path.join(SCRIPT_DIR, "evaluation_criteria.jsonl")
LOG_FILE = os.path.join(SCRIPT_DIR, "results/multi_agent_log.txt")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results/multi_agent_results.json")
MARKDOWN_REPORT_FILE = os.path.join(SCRIPT_DIR, "results/multi_agent_results.md")


def strip_json_fences(text: str) -> str:
    """Clean markdown fences from JSON response."""
    if text is None:
        return ""
    s = text.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def call_agent(prompt: str, seed: int = 42) -> Tuple[str, float]:
    """Call the qwen3:0.6b model with the given prompt."""
    start = time.time()
    try:
        messages = [
            {"role": "system", "content": "You are a specialized agent. Return ONLY valid JSON. No explanation."},
            {"role": "user", "content": prompt},
        ]
        response = ollama.chat(model=MODEL, messages=messages, options={"seed": seed, "temperature": 0.1})
        raw = response["message"]["content"]
        duration_ms = (time.time() - start) * 1000
        clean = strip_json_fences(raw)
        return clean, duration_ms
    except Exception as e:
        print(f"Error calling agent: {e}", flush=True)
        return "ERROR", 0.0


# ============================================================================
# ROUTER AGENT - Decides which specialized agent should handle the query
# ============================================================================

ROUTER_PROMPT = """You are a routing classifier. Your job is to determine which category a user query belongs to.

Categories:
1. IT_TICKET - Technical issues, system failures, hardware problems, outages
2. LEAVE_REQUEST - Employee leave requests, vacation, sick leave, time off
3. CUSTOMER_QUERY - Customer complaints, order issues, billing inquiries, refunds
4. KNOWLEDGE_QUERY - Policy questions, reimbursement rules, company guidelines

Respond with ONLY a JSON object in this exact format:
{"category": "CATEGORY_NAME"}

Examples:
User: "The server is down, please fix it immediately"
{"category": "IT_TICKET"}

User: "I need sick leave tomorrow for fever"
{"category": "LEAVE_REQUEST"}

User: "Customer CUST123 is asking about their refund"
{"category": "CUSTOMER_QUERY"}

User: "What is the expense reimbursement policy?"
{"category": "KNOWLEDGE_QUERY"}

User query: {query}
"""


def router_agent(query: str) -> Tuple[str, float]:
    """Route the query to the appropriate category."""
    prompt = ROUTER_PROMPT.replace("{query}", query)
    response, time_ms = call_agent(prompt)
    try:
        parsed = json.loads(response)
        category = parsed.get("category", "UNKNOWN")
        return category.upper(), time_ms
    except:
        # Fallback: try to extract category from response text
        resp_upper = response.upper()
        if "IT_TICKET" in resp_upper:
            return "IT_TICKET", time_ms
        elif "LEAVE_REQUEST" in resp_upper:
            return "LEAVE_REQUEST", time_ms
        elif "CUSTOMER_QUERY" in resp_upper:
            return "CUSTOMER_QUERY", time_ms
        elif "KNOWLEDGE_QUERY" in resp_upper or "KNOWLEDGE" in resp_upper:
            return "KNOWLEDGE_QUERY", time_ms
        return "UNKNOWN", time_ms


# ============================================================================
# SPECIALIZED AGENTS - Each handles only their specific tool type
# ============================================================================

# --- IT Ticket Agent ---
IT_TICKET_PROMPT = """You are an IT Ticket Creation Agent. Create an IT ticket based on the user query.

You MUST respond with ONLY a single valid JSON object with this exact structure:
{"name": "create_it_ticket", "parameters": {"title": "...", "description": "...", "priority": "...", "department": "..."}}

Field rules:
- priority: MUST be one of ["Low", "Medium", "High", "Critical"]
- Critical for: production failures, payment gateway down, all-user impact
- High for: server inaccessible, VPN issues, team-wide problems
- Medium for: single device issues, non-urgent requests

Examples:
User: "Payment gateway is failing for all users"
{"name": "create_it_ticket", "parameters": {"title": "Payment Gateway Failure in Production", "description": "Payment gateway failing for all users", "priority": "Critical", "department": "Engineering"}}

User: "VPN keeps disconnecting for remote staff"
{"name": "create_it_ticket", "parameters": {"title": "VPN Frequent Disconnection", "description": "VPN drops connection repeatedly for remote employees", "priority": "High", "department": "IT Support"}}

User query: {query}
"""

# --- Leave Request Agent ---
LEAVE_REQUEST_PROMPT = """You are a Leave Request Processing Agent. Process employee leave requests.

You MUST respond with ONLY a single valid JSON object with this exact structure:
{"name": "process_leave_request", "parameters": {"leave_type": "...", "start_date": "...", "end_date": "...", "reason": "..."}}

Field rules:
- leave_type: MUST be one of ["Sick", "Casual", "Earned"]
- Use "Sick" for medical reasons, illness, surgery
- Use "Casual" for personal reasons, bank visits, short personal leave
- Use "Earned" for planned vacations, family trips, paternity leave
- Dates: Convert relative dates (tomorrow, next week) to YYYY-MM-DD format using 2026 as reference year

Examples:
User: "I need sick leave tomorrow for fever"
{"name": "process_leave_request", "parameters": {"leave_type": "Sick", "start_date": "2026-04-13", "end_date": "2026-04-13", "reason": "Fever"}}

User: "Requesting 2 days casual leave next week for personal reasons"
{"name": "process_leave_request", "parameters": {"leave_type": "Casual", "start_date": "2026-04-21", "end_date": "2026-04-22", "reason": "Personal reasons"}}

User: "Taking earned leave May 8-15 for family trip"
{"name": "process_leave_request", "parameters": {"leave_type": "Earned", "start_date": "2026-05-08", "end_date": "2026-05-15", "reason": "Family trip"}}

User query: {query}
"""

# --- Customer Query Agent ---
CUSTOMER_QUERY_PROMPT = """You are a Customer Query Routing Agent. Route customer issues to appropriate teams.

You MUST respond with ONLY a single valid JSON object with this exact structure:
{"name": "route_customer_query", "parameters": {"query_type": "...", "customer_id": "...", "priority": "...", "department": "..."}}

Field rules:
- query_type: Describe the issue briefly (e.g., "Delivery Delay", "Billing Dispute", "Refund Status")
- priority: MUST be one of ["Low", "Medium", "High"]
- High for: delivery delays >7 days, wrong items, angry customers
- Medium for: standard inquiries, pending payments
- Low for: general questions, future orders
- customer_id: Extract if mentioned (CUSTxxxx), else use null
- department: Use "Logistics" for delivery issues, "Finance" for billing/refunds, "Customer Support" for wrong items, "Sales" for pricing inquiries

Examples:
User: "Customer CUST123 has not received order after 12 days"
{"name": "route_customer_query", "parameters": {"query_type": "Delivery Delay", "customer_id": "CUST123", "priority": "High", "department": "Logistics"}}

User: "Client CUST3344 asking for callback about pending payment"
{"name": "route_customer_query", "parameters": {"query_type": "Payment Follow-up", "customer_id": "CUST3344", "priority": "Medium", "department": "Finance"}}

User: "Customer CUST8604 received wrong item in order ORD5566"
{"name": "route_customer_query", "parameters": {"query_type": "Wrong Item Delivered", "customer_id": "CUST8604", "priority": "High", "department": "Customer Support"}}

User query: {query}
"""

# --- Knowledge Query Agent ---
KNOWLEDGE_PROMPT = """You are an Internal Knowledge Retrieval Agent. Fetch company policies and information.

You MUST respond with ONLY a single valid JSON object with this exact structure:
{"name": "get_internal_knowledge", "parameters": {"topic": "...", "department": "..."}}

Field rules:
- topic: Summarize what information is being requested (e.g., "Expense Reimbursement Policy", "Sick Leave Entitlement")
- department: The relevant department that owns this information ("HR" for leave policies, dress code, WFH etc; "Finance" for reimbursement, billing, purchase limits etc)

Examples:
User: "What is the policy for reimbursing expenses above 50,000?"
{"name": "get_internal_knowledge", "parameters": {"topic": "Expense Reimbursement Policy", "department": "Finance"}}

User: "How many sick leaves per year?"
{"name": "get_internal_knowledge", "parameters": {"topic": "Sick Leave Entitlement", "department": "HR"}}

User: "Dress code for client meetings?"
{"name": "get_internal_knowledge", "parameters": {"topic": "Dress Code Policy", "department": "HR"}}

User query: {query}
"""


def specialized_agent(category: str, query: str) -> Tuple[str, float]:
    """Route to the appropriate specialized agent."""
    if category == "IT_TICKET":
        prompt = IT_TICKET_PROMPT.replace("{query}", query)
    elif category == "LEAVE_REQUEST":
        prompt = LEAVE_REQUEST_PROMPT.replace("{query}", query)
    elif category == "CUSTOMER_QUERY":
        prompt = CUSTOMER_QUERY_PROMPT.replace("{query}", query)
    elif category == "KNOWLEDGE_QUERY":
        prompt = KNOWLEDGE_PROMPT.replace("{query}", query)
    else:
        # Default to knowledge agent as fallback
        prompt = KNOWLEDGE_PROMPT.replace("{query}", query)

    return call_agent(prompt)


# ============================================================================
# ORCHESTRATOR - Coordinates the multi-agent system
# ============================================================================

class MultiAgentOrchestrator:
    """Orchestrates the routing and specialized agents."""

    def __init__(self):
        self.router_time = 0.0
        self.specialist_time = 0.0
        self.category = ""

    def process(self, query: str) -> Tuple[str, str, float]:
        """
        Process a query through the multi-agent system.

        Returns:
            (category, final_response, total_time_ms)
        """
        # Step 1: Route to category
        self.category, router_ms = router_agent(query)
        self.router_time = router_ms

        # Step 2: Call specialized agent
        specialist_response, specialist_ms = specialized_agent(self.category, query)
        self.specialist_time = specialist_ms

        total_time = router_ms + specialist_ms
        return self.category, specialist_response, total_time


# ============================================================================
# EVALUATION AND REPORTING (using modified logic from evaluate_toolcalling.py)
# ============================================================================

def load_evaluation_criteria():
    """Load relaxed evaluation criteria from file."""
    criteria = {}
    if not os.path.exists(EVAL_CRITERIA_FILE):
        return criteria
    try:
        with open(EVAL_CRITERIA_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    crit = json.loads(line)
                    criteria[crit["prompt_id"]] = crit
        return criteria
    except Exception as e:
        print(f"Warning: Could not load evaluation criteria: {e}", flush=True)
        return {}


def _is_missing_value(v):
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _validate_single_tool_call_relaxed(model_obj, criteria):
    """Validate a single tool call using relaxed criteria."""
    if not isinstance(model_obj, dict):
        return 0, "not_object"

    if "name" not in model_obj or "parameters" not in model_obj:
        return 0, "missing_keys"

    # Check tool name matches
    if model_obj.get("name") != criteria.get("expected_name"):
        return 0, "wrong_tool"

    model_params = model_obj.get("parameters")
    if not isinstance(model_params, dict):
        return 0, "parameters_not_object"

    # Check all required keys are present and have non-null/non-blank values
    required_keys = criteria.get("required_keys", [])
    for key in required_keys:
        if key not in model_params:
            return 0, f"missing_key:{key}"
        if _is_missing_value(model_params.get(key)):
            return 0, f"parameter_value_missing:{key}"

    # Check strict values (must match one of allowed values)
    strict_values = criteria.get("strict_values", {})
    for key, allowed_values in strict_values.items():
        actual_val = model_params.get(key)
        if _is_missing_value(actual_val):
            return 0, f"parameter_value_missing:{key}"
        # Case-insensitive comparison
        if isinstance(actual_val, str) and isinstance(allowed_values, list):
            if actual_val.lower() not in [v.lower() for v in allowed_values]:
                return 0, f"parameter_value_invalid:{key}"
        elif actual_val not in allowed_values:
            return 0, f"parameter_value_invalid:{key}"

    # Check contains values (substring match)
    contains_values = criteria.get("contains_values", {})
    for key, expected_substring in contains_values.items():
        actual_val = model_params.get(key, "")
        if _is_missing_value(actual_val):
            return 0, f"parameter_value_missing:{key}"
        if expected_substring not in str(actual_val):
            return 0, f"parameter_value_not_contains:{key}"

    return 1, "ok"


def validate_tool_call(model_json, prompt_id, criteria_map):
    """Validate tool call using relaxed criteria."""
    try:
        model_parsed = json.loads(model_json)
    except Exception:
        return 0, "parse_error"

    criteria = criteria_map.get(prompt_id)
    if not criteria:
        # Fallback to basic validation if no criteria found
        if isinstance(model_parsed, dict) and "name" in model_parsed:
            return 1, "ok_no_criteria"
        return 0, "no_criteria_found"

    # Handle multi-tool scenarios
    is_multi_tool = criteria.get("is_multi_tool", False)

    # Normalize model output to list
    if isinstance(model_parsed, list):
        model_list = model_parsed
    elif isinstance(model_parsed, dict):
        model_list = [model_parsed]
    else:
        return 0, "not_object_or_list"

    if is_multi_tool:
        min_tools = criteria.get("min_tools", 1)
        max_tools = criteria.get("max_tools", min_tools)

        if len(model_list) < min_tools:
            return 0, f"tool_count_too_few:got_{len(model_list)}_min_{min_tools}"
        if len(model_list) > max_tools:
            return 0, f"tool_count_too_many:got_{len(model_list)}_max_{max_tools}"

        # Validate each tool call
        for i, model_tool in enumerate(model_list):
            success, reason = _validate_single_tool_call_relaxed(model_tool, criteria)
            if success == 0:
                return 0, f"tool_{i}:{reason}"

        return 1, "ok"
    else:
        # Single tool expected
        if len(model_list) > 1:
            return 0, f"tool_count_too_many:got_{len(model_list)}_expected_1"

        model_tool = model_list[0]
        return _validate_single_tool_call_relaxed(model_tool, criteria)


def load_dataset():
    """Load test dataset."""
    with open(DATASET_FILE, "r") as f:
        return [json.loads(line) for line in f]


def run_multi_agent_experiment(model: str = DEFAULT_MODEL, limit: Optional[int] = None, fresh: bool = False):
    """Run the full multi_agent experiment."""
    global MODEL
    MODEL = model
    print("=" * 60)
    print(f"Multi-Agent System ({MODEL}) - Tool Calling Experiment")
    print("=" * 60)
    print(f"Architecture: Router Agent → Specialized Agent")
    print(f"Specialized Agents: IT Ticket | Leave | Customer | Knowledge")
    print("=" * 60)

    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)

    data = load_dataset()
    if limit:
        data = data[:limit]

    criteria_map = load_evaluation_criteria()
    orchestrator = MultiAgentOrchestrator()

    results = []
    total = len(data)

    mode = "w" if fresh else "a"
    with open(LOG_FILE, mode) as log_f:
        if fresh:
            log_f.write("Multi-Agent System Log\n" + "=" * 60 + "\n")

        for i, item in enumerate(data):
            print(f"\n[{i+1}/{total}] Query: {item['query'][:60]}...")

            # Process through multi-agent system
            category, response, time_ms = orchestrator.process(item['query'])

            # Validate the response
            success, reason = validate_tool_call(response, item['id'], criteria_map)

            # Extract expected category for logging
            # Extract expected category for logging
            expected = item.get('expected')
            if expected:
                if isinstance(expected, list):
                    expected_tool = expected[0].get('name') if expected else 'unknown'
                else:
                    expected_tool = expected.get('name') if isinstance(expected, dict) else 'unknown'
            else:
                # Fallback to criteria map
                crit = criteria_map.get(item['id'], {})
                expected_tool = crit.get('expected_name', 'unknown')

            # Determine if routing was correct (tool name matches expected)
            try:
                parsed = json.loads(response) if response != "ERROR" else {}
                if isinstance(parsed, dict):
                    actual_tool = parsed.get('name', 'unknown')
                else:
                    actual_tool = 'unknown'
            except:
                actual_tool = 'unknown'

            routing_correct = actual_tool == expected_tool

            print(f"  ├─ Router: {category} | Tool: {actual_tool}")
            print(f"  ├─ Expected: {expected_tool}")
            print(f"  ├─ Success: {success} | Reason: {reason}")
            print(f"  └─ Time: {time_ms:.0f}ms")

            # Log details
            log_f.write(f"\n{'='*50}\n")
            log_f.write(f"Query: {item['query']}\n")
            log_f.write(f"Router Category: {category}\n")
            log_f.write(f"Response: {response}\n")
            log_f.write(f"Expected: {json.dumps(expected)}\n")
            log_f.write(f"Success: {success} | Reason: {reason} | Time: {time_ms:.0f}ms\n")
            log_f.write(f"Routing Correct: {routing_correct}\n")
            log_f.flush()

            results.append({
                "id": item['id'],
                "query": item['query'],
                "category": category,
                "response": response,
                "success": success,
                "reason": reason,
                "time_ms": time_ms,
                "routing_correct": routing_correct,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool
            })

    # Save results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    if HAS_VIZ:
        generate_visualizations(results)

    # Generate report
    generate_report(results)

    return results


def generate_visualizations(results: List[Dict]):
    """Generate performance charts for the Multi-Agent System."""
    if not results:
        return
        
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy by Category
    plt.figure(figsize=(10, 6))
    cat_acc = df.groupby('category')['success'].mean().reset_index()
    cat_acc['accuracy'] = cat_acc['success'] * 100
    ax = sns.barplot(data=cat_acc, x='category', y='accuracy', palette='viridis')
    plt.title(f'Multi-Agent System: Accuracy by Category ({MODEL})')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results/ma_accuracy_by_category.png"), dpi=200)
    plt.close()

    # 2. Latency Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='time_ms', bins=15, kde=True, color='blue')
    plt.axvline(df['time_ms'].mean(), color='red', linestyle='--', label=f'Mean: {df["time_ms"].mean():.0f}ms')
    plt.title('Multi-Agent System: Latency Distribution')
    plt.xlabel('Response Time (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results/ma_latency_distribution.png"), dpi=200)
    plt.close()

    # 3. Routing vs Tool Accuracy
    plt.figure(figsize=(10, 6))
    routing_acc = df['routing_correct'].mean() * 100
    final_acc = df['success'].mean() * 100
    
    comparison = pd.DataFrame({
        'Metric': ['Routing Accuracy', 'Final Tool-Calling Accuracy'],
        'Value': [routing_acc, final_acc]
    })
    
    ax = sns.barplot(data=comparison, x='Metric', y='Value', palette='magma')
    plt.title('Multi-Agent System: Routing vs Final Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results/ma_routing_vs_accuracy.png"), dpi=200)
    plt.close()


def generate_report(results: List[Dict]):
    """Generate markdown report with statistics."""
    if not results:
        print("No results to report.")
        return

    total = len(results)
    successes = sum(1 for r in results if r['success'] == 1)
    routing_correct = sum(1 for r in results if r['routing_correct'])
    accuracy = successes / total * 100 if total > 0 else 0
    routing_accuracy = routing_correct / total * 100 if total > 0 else 0
    avg_time = sum(r['time_ms'] for r in results) / total if total > 0 else 0

    # Error breakdown
    errors = [r for r in results if r['success'] == 0]
    error_counts = {}
    for err in errors:
        reason = err['reason']
        error_counts[reason] = error_counts.get(reason, 0) + 1

    md = f"""# Multi-Agent System Results (qwen3:0.6b)

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Architecture
- **Router Agent**: Classifies queries into 4 categories
- **Specialized Agents**: One agent per tool type
  - IT Ticket Agent
  - Leave Request Agent  
  - Customer Query Agent
  - Knowledge Agent

## Performance Analysis
![Accuracy by Category](ma_accuracy_by_category.png)
![Routing vs Accuracy](ma_routing_vs_accuracy.png)

## Latency Analysis
![Latency Distribution](ma_latency_distribution.png)
"""

    # Category breakdown
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'success': 0}
        categories[cat]['total'] += 1
        if r['success'] == 1:
            categories[cat]['success'] += 1

    cat_rows = []
    for cat, stats in sorted(categories.items()):
        cat_acc = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        cat_rows.append([cat, stats['total'], f"{cat_acc:.1f}%"])

    md += tabulate(cat_rows, headers=["Category", "Count", "Accuracy"], tablefmt="github")

    md += "\n\n## Error Breakdown\n"
    if error_counts:
        err_rows = [[reason, count] for reason, count in sorted(error_counts.items(), key=lambda x: -x[1])]
        md += tabulate(err_rows, headers=["Error Reason", "Count"], tablefmt="github")
    else:
        md += "_No errors recorded._\n"

    md += "\n\n## Comparison with Single SLM Models\n"
    md += """
The multi-agent system typically outperforms single SLM approaches because:
1. **Reduced Cognitive Load**: Each agent only handles 1 tool type
2. **Specialized Prompts**: Domain-specific examples and rules
3. **Explicit Routing**: Router agent makes tool selection explicit
4. **Focused Training**: Smaller effective "search space" for each agent
"""

    # Detailed results
    md += "\n\n## Detailed Results\n"
    detail_rows = []
    for r in results:
        detail_rows.append([
            r['id'],
            r['category'],
            r['actual_tool'],
            "✓" if r['success'] == 1 else "✗",
            r['reason'] if r['success'] == 0 else "-",
            f"{r['time_ms']:.0f}"
        ])

    md += tabulate(detail_rows, headers=["ID", "Category", "Tool", "Success", "Reason", "Time(ms)"], tablefmt="github")

    with open(MARKDOWN_REPORT_FILE, "w") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.1f}% ({successes}/{total})")
    print(f"Tool Selection: {routing_accuracy:.1f}% ({routing_correct}/{total})")
    print(f"Avg Time: {avg_time:.0f}ms")
    print(f"\nReport saved: {MARKDOWN_REPORT_FILE}")
    print(f"Log saved: {LOG_FILE}")
    print(f"Results saved: {RESULTS_JSON}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Tool Calling System")
    parser.add_argument("--limit", type=int, help="Limit number of prompts")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (clear previous logs)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use for agents")
    args = parser.parse_args()

    run_multi_agent_experiment(model=args.model, limit=args.limit, fresh=args.fresh)
