import argparse
import json
import os
import re
import time

import ollama
import pandas as pd
import requests
from tabulate import tabulate

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_VIZ = True
except Exception:
    HAS_VIZ = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_MODELS = [
    "qwen3:0.6b",
    "deepseek-r1:7b",
    "phi4:latest"
]

MODEL_SIZES = {
    "qwen3:0.6b": 0.6,
    "deepseek-r1:7b": 7.0,
    "phi4:latest": 14.0
}

DATASET_FILE = os.path.join(SCRIPT_DIR, "prompts.jsonl")
EVAL_CRITERIA_FILE = os.path.join(SCRIPT_DIR, "evaluation_criteria.jsonl")
LOG_FILE = os.path.join(SCRIPT_DIR, "results/detailed_toolcalling_log.txt")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results/results_data.json")
MARKDOWN_REPORT_FILE = os.path.join(SCRIPT_DIR, "results/results.md")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "results")

ACCURACY_PNG = os.path.join(SCRIPT_DIR,"results/accuracy_comparison.png")
TIMING_PNG = os.path.join(SCRIPT_DIR,"results/timing_comparison.png")
ERRORS_PNG = os.path.join(SCRIPT_DIR,"results/error_breakdown.png")

ZERO_SHOT_PROMPT = """
You are an enterprise tool-calling agent. 

You MUST respond with ONLY a single valid JSON object. 
Do not add any explanation, markdown, or extra text before or after the JSON.

Use this exact structure for response:

{
  "name": "tool_name",
  "parameters": {
    "parameter_name": "value"
  }
}

Never invent new keys. Use only the parameter names defined in the tool description.
######Available tools############
{
  "create_it_ticket": {
    "name": "create_it_ticket",
    "parameters": {
      "type": "object",
      "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "priority": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
        "department": {"type": "string"}
      },
      "required": ["title", "description", "priority", "department"]
    }
  },
  "process_leave_request": {
    "name": "process_leave_request",
    "parameters": {
      "type": "object",
      "properties": {
        "leave_type": {"type": "string", "enum": ["Sick", "Casual", "Earned"]},
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "reason": {"type": "string"}
      },
      "required": ["leave_type", "start_date", "end_date", "reason"]
    }
  },
  "route_customer_query": {
    "name": "route_customer_query",
    "parameters": {
      "type": "object",
      "properties": {
        "query_type": {"type": "string"},
        "customer_id": {"type": "string"},
        "priority": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "department": {"type": "string"}
      },
      "required": ["query_type", "priority", "department"]
    }
  },
  "get_internal_knowledge": {
    "name": "get_internal_knowledge",
    "parameters": {
      "type": "object",
      "properties": {
        "topic": {"type": "string"},
        "department": {"type": "string"}
      },
      "required": ["topic", "department"]
    }
  }
}

User query: {query}
"""

FEW_SHOT_PROMPT = """
You are an enterprise tool-calling agent. 

You MUST respond with ONLY a single valid JSON object. 
Do not add any explanation, markdown, or extra text before or after the JSON.

Use this exact structure for response:

{
  "name": "tool_name",
  "parameters": {
    "parameter_name": "value"
  }
}

Never invent new keys. Use only the parameter names defined in the tool description.
######Available tools############
{
  "create_it_ticket": {
    "name": "create_it_ticket",
    "parameters": {
      "type": "object",
      "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "priority": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
        "department": {"type": "string"}
      },
      "required": ["title", "description", "priority", "department"]
    }
  },
  "process_leave_request": {
    "name": "process_leave_request",
    "parameters": {
      "type": "object",
      "properties": {
        "leave_type": {"type": "string", "enum": ["Sick", "Casual", "Earned"]},
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "reason": {"type": "string"}
      },
      "required": ["leave_type", "start_date", "end_date", "reason"]
    }
  },
  "route_customer_query": {
    "name": "route_customer_query",
    "parameters": {
      "type": "object",
      "properties": {
        "query_type": {"type": "string"},
        "customer_id": {"type": "string"},
        "priority": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "department": {"type": "string"}
      },
      "required": ["query_type", "priority", "department"]
    }
  },
  "get_internal_knowledge": {
    "name": "get_internal_knowledge",
    "parameters": {
      "type": "object",
      "properties": {
        "topic": {"type": "string"},
        "department": {"type": "string"}
      },
      "required": ["topic", "department"]
    }
  }
}

Examples:
{"name": "create_it_ticket", "parameters": {"title": "Lift problem", "description": "Lift stuck on 19th floor and showing error code ESP", "priority": "Medium", "department": "Office Admin"}}
{"name": "process_leave_request", "parameters": {"leave_type": "Casual", "start_date": "2026-03-20", "end_date": "2026-03-22", "reason": "Durga Puja"}}
{"name": "route_customer_query", "parameters": {"query_type": "Billing Dispute", "customer_id": "CUST567", "priority": "High", "department": "Finance"}}
{"name": "get_internal_knowledge", "parameters": {"topic": "Company Leave Policy", "department": "HR"}}

User query: {query}
"""

def unload_model(model_name):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model_name, "keep_alive": 0}
        requests.post(url, json=payload, timeout=10)
        print(f"Successfully unloaded {model_name}", flush=True)
    except Exception as e:
        print(f"Error unloading {model_name}: {e}", flush=True)


def load_existing_results():
    if not os.path.exists(RESULTS_JSON):
        return []
    try:
        with open(RESULTS_JSON, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except Exception:
        return []


def load_dataset():
    """Load test dataset."""
    with open(DATASET_FILE, "r") as f:
        return [json.loads(line) for line in f]


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


def strip_json_fences(text):
    if text is None:
        return ""
    s = text.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

    # Remove ```json ... ``` or ``` ... ``` wrappers
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    return s.strip()


def get_response(model, prompt, seed=None):
    start = time.time()
    try:
        options = {}
        if seed is not None:
            options["seed"] = seed

        messages = [
            {"role": "system", "content": "Return ONLY a single valid JSON object with 'name' and 'parameters' keys. No prose. No markdown."},
            {"role": "user", "content": prompt},
        ]
        raw = ollama.chat(model=model, messages=messages, options=options)["message"]["content"]
        duration_ms = (time.time() - start) * 1000
        clean = strip_json_fences(raw)
        return raw, clean, duration_ms
    except Exception as e:
        
        print(f"Error with {model}: {e}", flush=True)
        return "ERROR", "ERROR", 0.0


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
    """Validate tool call using relaxed criteria from evaluation_criteria file.

    Args:
        model_json: The raw JSON response from the model
        prompt_id: The prompt ID to look up criteria
        criteria_map: Dictionary mapping prompt_id to criteria
    """
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


def get_size(name):
    return MODEL_SIZES.get(name, 999.0)


def get_category_from_tool(tool_name):
    """Map tool name to category."""
    tool_to_category = {
        "create_it_ticket": "IT_TICKET",
        "process_leave_request": "LEAVE_REQUEST",
        "route_customer_query": "CUSTOMER_QUERY",
        "get_internal_knowledge": "KNOWLEDGE_QUERY"
    }
    return tool_to_category.get(tool_name, "UNKNOWN")


def log_verbose(file_handle, model, item, condition, prompt, raw, clean, success, reason, ms, expected_name="N/A"):
    file_handle.write(f"\n{'='*50}\nMODEL: {model} | ID: {item['id']} | COND: {condition}\n")
    file_handle.write(f"--- QUERY ---\n{item['query']}\n")
    file_handle.write(f"--- PROMPT ---\n{prompt}\n")
    file_handle.write(f"--- EXPECTED ---\n{expected_name}\n")
    file_handle.write(f"--- RESPONSE (Raw) ---\n{raw}\n")
    file_handle.write(f"--- RESPONSE (Clean) ---\n{clean}\n")
    file_handle.write(f"SUCCESS: {success} | REASON: {reason} | TIME_MS: {ms:.0f}\n")
    file_handle.write(f"{'='*50}\n")
    file_handle.flush()


def generate_unified_report(slm_results, multi_agent_results):
    """Generate unified report with SLM and Multi-Agent comparison."""
    md = f"# Tool Calling Results: SLM Models vs Multi-Agent System\nUpdated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md += "## Architecture Comparison\n\n"
    md += "| Approach | Description |\n"
    md += "|----------|-------------|\n"
    md += "| **SLM (Single Model)** | One model handles all 4 tools with full schema |\n"
    md += "| **Multi-Agent** | Router + 4 specialized agents (qwen3:0.6b each) |\n"
    md += "\n"
    
    md += "## Visualizations\n\n"
    md += "![Accuracy Comparison](accuracy_comparison.png)\n"
    md += "![Latency Comparison](timing_comparison.png)\n"
    md += "![Error Breakdown](error_breakdown.png)\n\n"

    # Load multi-agent results
    if multi_agent_results:
        ma_total = len(multi_agent_results)
        ma_success = sum(1 for r in multi_agent_results if r.get('success', 0) == 1)
        ma_accuracy = ma_success / ma_total * 100 if ma_total > 0 else 0
        ma_avg_time = sum(r.get('time_ms', 0) for r in multi_agent_results) / ma_total if ma_total > 0 else 0
        ma_routing = sum(1 for r in multi_agent_results if r.get('routing_correct', False)) / ma_total * 100 if ma_total > 0 else 0
    else:
        ma_accuracy, ma_avg_time, ma_routing = 0, 0, 0

    # SLM summary from results
    if slm_results:
        df = pd.DataFrame(slm_results)
        
        # Calculate per-model metrics
        model_summary = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            for condition in ['zero_shot', 'few_shot']:
                cond_df = model_df[model_df['condition'] == condition]
                if len(cond_df) > 0:
                    accuracy = cond_df['success'].mean() * 100
                    avg_time = cond_df['time_ms'].mean()
                    errors = len(cond_df[cond_df['success'] == 0])
                    error_breakdown = cond_df[cond_df['success'] == 0]['reason'].value_counts().to_dict()
                    model_summary.append({
                        'model': model,
                        'condition': condition,
                        'accuracy': accuracy,
                        'avg_time_ms': avg_time,
                        'total': len(cond_df),
                        'success': cond_df['success'].sum(),
                        'errors': errors,
                        'error_breakdown': error_breakdown
                    })
    else:
        model_summary = []

    # Combined Accuracy & Timing Comparison Table
    md += "## Accuracy & Timing Comparison\n\n"
    
    rows = []
    # Add SLM models
    for m in model_summary:
        cond_label = "ZS" if m['condition'] == 'zero_shot' else "FS"
        err_str = ", ".join([f"{k}:{v}" for k, v in list(m['error_breakdown'].items())[:3]]) if m['error_breakdown'] else "-"
        rows.append([
            f"{m['model']} ({cond_label})",
            f"{m['accuracy']:.1f}%",
            f"{m['avg_time_ms']:.0f}ms",
            f"{m['success']}/{m['total']}",
            err_str
        ])
    
    # Add Multi-Agent
    if multi_agent_results:
        ma_err_breakdown = {}
        for r in multi_agent_results:
            if r.get('success', 1) == 0:
                reason = r.get('reason', 'unknown')
                ma_err_breakdown[reason] = ma_err_breakdown.get(reason, 0) + 1
        ma_err_str = ", ".join([f"{k}:{v}" for k, v in ma_err_breakdown.items()]) if ma_err_breakdown else "-"
        rows.append([
            "**Multi-Agent (qwen3:0.6b)**",
            f"{ma_accuracy:.1f}%",
            f"{ma_avg_time:.0f}ms",
            f"{ma_success}/{ma_total}",
            ma_err_str
        ])
    
    md += tabulate(rows, headers=["Model/Approach", "Accuracy", "Avg Time", "Success/Total", "Key Errors"], tablefmt="github")
    md += "\n\n"
    
    # Category breakdown for Multi-Agent
    if multi_agent_results:
        md += "### Multi-Agent Accuracy by Category\n\n"
        cat_stats = {}
        for r in multi_agent_results:
            cat = r.get('category', 'UNKNOWN')
            if cat not in cat_stats:
                cat_stats[cat] = {'total': 0, 'success': 0}
            cat_stats[cat]['total'] += 1
            if r.get('success', 0) == 1:
                cat_stats[cat]['success'] += 1
        
        cat_rows = []
        for cat, stats in sorted(cat_stats.items()):
            acc = stats['success'] / stats['total'] * 100
            cat_rows.append([cat, stats['total'], f"{acc:.1f}%"])
        
        md += tabulate(cat_rows, headers=["Category", "Count", "Accuracy"], tablefmt="github")
        md += "\n\n"

    # Error Analysis Summary
    md += "## Error Analysis Summary\n\n"
    
    # Collect all errors
    all_errors = {}
    
    # SLM errors
    for m in model_summary:
        for err_type, count in m['error_breakdown'].items():
            key = f"{m['model']} ({m['condition']})"
            if key not in all_errors:
                all_errors[key] = {}
            all_errors[key][err_type] = count
    
    # Multi-agent errors
    if multi_agent_results:
        ma_errors = {}
        for r in multi_agent_results:
            if r.get('success', 1) == 0:
                reason = r.get('reason', 'unknown')
                ma_errors[reason] = ma_errors.get(reason, 0) + 1
        if ma_errors:
            all_errors["Multi-Agent"] = ma_errors
    
    if all_errors:
        # Get unique error types
        error_types = set()
        for errs in all_errors.values():
            error_types.update(errs.keys())
        error_types = sorted(error_types)
        
        # Build table
        err_rows = []
        for model_key, errs in sorted(all_errors.items()):
            row = [model_key] + [errs.get(et, 0) for et in error_types] + [sum(errs.values())]
            err_rows.append(row)
        
        headers = ["Model"] + error_types + ["Total"]
        md += tabulate(err_rows, headers=headers, tablefmt="github")
        md += "\n\n"
    else:
        md += "_No errors recorded._\n\n"
    
    # Performance Insights
    md += "## Performance Insights\n\n"
    
    if model_summary and multi_agent_results:
        best_slm = max(model_summary, key=lambda x: x['accuracy'])
        md += f"- **Best SLM**: {best_slm['model']} ({best_slm['condition']}) at {best_slm['accuracy']:.1f}% accuracy\n"
        md += f"- **Multi-Agent**: {ma_accuracy:.1f}% accuracy with {ma_avg_time:.0f}ms avg latency\n"
        
        if ma_accuracy > best_slm['accuracy']:
            improvement = ma_accuracy - best_slm['accuracy']
            md += f"- **Multi-Agent outperforms best SLM by {improvement:.1f} percentage points**\n"
        
        md += "\n### Why Multi-Agent Works Better\n\n"
        md += "1. **Reduced Cognitive Load**: Each agent only handles 1 tool type\n"
        md += "2. **Specialized Prompts**: Domain-specific examples and rules per agent\n"
        md += "3. **Explicit Routing**: Router agent makes tool selection a separate, focused decision\n"
        md += "4. **Smaller Search Space**: Each agent has fewer valid outputs to consider\n"
    
    with open(MARKDOWN_REPORT_FILE, "w") as f:
        f.write(md)
    
    print(f"\nUnified report saved: {MARKDOWN_REPORT_FILE}")


def generate_markdown_report_incremental(results_list, models_list):
    # Store results in multi-agent format for unified reporting
    structured_results = []
    for r in results_list:
        # Get expected tool from stored expected output
        expected_tool = None
        if 'expected_tool' in r:
            expected_tool = r['expected_tool']
        else:
            # Try to extract from query by loading dataset
            try:
                dataset = load_dataset()
                item = next((d for d in dataset if d['id'] == r['id']), None)
                if item:
                    expected = item['expected']
                    if isinstance(expected, list):
                        expected_tool = expected[0].get('name') if expected else 'unknown'
                    else:
                        expected_tool = expected.get('name') if isinstance(expected, dict) else 'unknown'
            except:
                expected_tool = 'unknown'
        
        # Get actual tool from response
        actual_tool = None
        routing_correct = False
        try:
            parsed = json.loads(r.get('clean', '{}'))
            if isinstance(parsed, dict):
                actual_tool = parsed.get('name', 'unknown')
            else:
                actual_tool = 'unknown'
        except:
            actual_tool = 'unknown'
        
        if expected_tool and actual_tool:
            routing_correct = actual_tool == expected_tool
        
        category = get_category_from_tool(expected_tool) if expected_tool else 'UNKNOWN'
        
        structured_results.append({
            'id': r['id'],
            'query': r.get('query', ''),
            'category': category,
            'response': r.get('clean', ''),
            'success': r['success'],
            'reason': r['reason'],
            'time_ms': r['time_ms'],
            'routing_correct': routing_correct,
            'expected_tool': expected_tool or 'unknown',
            'actual_tool': actual_tool,
            'model': r.get('model', ''),
            'condition': r.get('condition', '')
        })
    
    # Save structured results
    with open(RESULTS_JSON, "w") as f:
        json.dump(structured_results, f, indent=2)
    
    # Load multi-agent results if available
    multi_agent_results = []
    ma_file = os.path.join(SCRIPT_DIR, "results/multi_agent_results.json")
    if os.path.exists(ma_file):
        try:
            with open(ma_file, "r") as f:
                multi_agent_results = json.load(f)
        except:
            pass
    
    # Generate visualizations
    generate_visualizations(structured_results, models_list)
    
    # Generate unified report
    generate_unified_report(structured_results, multi_agent_results)


def generate_visualizations(results_list, models_list):
    if not HAS_VIZ:
        return
    if not results_list:
        return

    df = pd.DataFrame(results_list)
    df = df[df["model"].isin(models_list)].copy()
    if df.empty:
        return

    sorted_models = sorted(models_list, key=get_size)
    df["model"] = pd.Categorical(df["model"], categories=sorted_models, ordered=True)

    # Load multi-agent results if available for comparison
    ma_file = os.path.join(SCRIPT_DIR, "results/multi_agent_results.json")
    if os.path.exists(ma_file):
        try:
            with open(ma_file, "r") as f:
                ma_results = json.load(f)
                if ma_results:
                    ma_df = pd.DataFrame(ma_results)
                    ma_acc = ma_df["success"].mean()
                    # Add a dummy model entry for Multi-Agent
                    ma_entry = pd.DataFrame({
                        "model": ["Multi-Agent (qwen3:0.6b)"],
                        "condition": ["multi_agent"],
                        "accuracy": [ma_acc]
                    })
        except:
            ma_entry = None
    else:
        ma_entry = None

    sns.set_theme(style="whitegrid")

    acc = (
        df.groupby(["model", "condition"])["success"].mean().reset_index().rename(columns={"success": "accuracy"})
    )
    
    if ma_entry is not None:
        acc = pd.concat([acc, ma_entry], ignore_index=True)
        acc["model"] = pd.Categorical(acc["model"], categories=list(sorted_models) + ["Multi-Agent (qwen3:0.6b)"], ordered=True)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=acc, x="model", y="accuracy", hue="condition")
    ax.set_title("Accuracy Comparison: SLM Models vs Multi-Agent")
    ax.set_xlabel("Model / Approach")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(ACCURACY_PNG, dpi=200)
    plt.close()

    # Timing comparison as line chart
    tms = df.groupby(["model", "condition"])["time_ms"].mean().reset_index()
    plt.figure(figsize=(12, 5))
    
    for condition in ["zero_shot", "few_shot"]:
        cond_data = tms[tms["condition"] == condition]
        if not cond_data.empty:
            marker = "o" if condition == "zero_shot" else "s"
            linestyle = "-" if condition == "zero_shot" else "--"
            label = "Zero-shot" if condition == "zero_shot" else "Few-shot"
            plt.plot(cond_data["model"], cond_data["time_ms"], marker=marker, linestyle=linestyle, label=label, linewidth=2, markersize=8)
    
    plt.title("Avg Latency (ms): Zero-shot vs Few-shot")
    plt.xlabel("Model")
    plt.ylabel("Avg time (ms)")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TIMING_PNG, dpi=200)
    plt.close()

    err = df[df["success"] == 0].copy()
    if err.empty:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No errors recorded.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(ERRORS_PNG, dpi=200)
        plt.close()
        return

    # Create 3 subplots for error breakdown
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Chart 1: Overall error types by condition (original)
    err_counts = (
        err.groupby(["condition", "reason"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    top_reasons = err_counts.groupby("reason")["count"].sum().sort_values(ascending=False).head(8).index.tolist()
    err_counts["reason"] = err_counts["reason"].where(err_counts["reason"].isin(top_reasons), other="other")
    err_counts = err_counts.groupby(["condition", "reason"], as_index=False)["count"].sum()
    
    ax1 = axes[0]
    sns.barplot(data=err_counts, x="reason", y="count", hue="condition", ax=ax1)
    ax1.set_title("Error Types (count) by Condition (Overall)")
    ax1.set_xlabel("Error reason")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=30)
    
    # Chart 2: Zero-shot errors by model
    ax2 = axes[1]
    zero_err = err[err["condition"] == "zero_shot"].copy()
    if not zero_err.empty:
        zero_err_counts = (
            zero_err.groupby(["model", "reason"]).size().reset_index(name="count").sort_values("count", ascending=False)
        )
        # Get top reasons
        top_z_reasons = zero_err_counts.groupby("reason")["count"].sum().sort_values(ascending=False).head(8).index.tolist()
        zero_err_counts["reason"] = zero_err_counts["reason"].where(zero_err_counts["reason"].isin(top_z_reasons), other="other")
        zero_err_counts = zero_err_counts.groupby(["model", "reason"], as_index=False)["count"].sum()
        sns.barplot(data=zero_err_counts, x="model", y="count", hue="reason", ax=ax2)
    ax2.set_title("Zero-shot: Error Types by Model")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=30)
    
    # Chart 3: Few-shot errors by model
    ax3 = axes[2]
    few_err = err[err["condition"] == "few_shot"].copy()
    if not few_err.empty:
        few_err_counts = (
            few_err.groupby(["model", "reason"]).size().reset_index(name="count").sort_values("count", ascending=False)
        )
        # Get top reasons
        top_f_reasons = few_err_counts.groupby("reason")["count"].sum().sort_values(ascending=False).head(8).index.tolist()
        few_err_counts["reason"] = few_err_counts["reason"].where(few_err_counts["reason"].isin(top_f_reasons), other="other")
        few_err_counts = few_err_counts.groupby(["model", "reason"], as_index=False)["count"].sum()
        sns.barplot(data=few_err_counts, x="model", y="count", hue="reason", ax=ax3)
    ax3.set_title("Few-shot: Error Types by Model")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Count")
    ax3.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(ERRORS_PNG, dpi=200)
    plt.close()


def run_experiment(models=None, limit=None, fresh=False, seed=None, max_size=None):
    print("Starting Experiment 12 (Tool Calling: Zero-shot vs Few-shot)...", flush=True)
    os.makedirs("results", exist_ok=True)

    models_to_test = models if models else ALL_MODELS
    if max_size:
        models_to_test = [m for m in models_to_test if get_size(m) < max_size]

    with open(DATASET_FILE, "r") as f:
        data = [json.loads(line) for line in f]

    if limit:
        data = data[:limit]

    results_list = [] if fresh else load_existing_results()

    mode = "w" if fresh else "a"
    with open(LOG_FILE, mode) as log_f:
        if fresh:
            log_f.write("Experiment 12 Tool Calling Log\n" + "=" * 40 + "\n")

        completed_pairs = set((r["model"], r["condition"]) for r in results_list)

        for model in models_to_test:
            for condition in ["zero_shot", "few_shot"]:
                if (model, condition) in completed_pairs:
                    continue

                # Load evaluation criteria
                criteria_map = load_evaluation_criteria()

                print(f"Testing Model: {model} | Condition: {condition}", flush=True)
                for item in data:
                    prompt_template = ZERO_SHOT_PROMPT if condition == "zero_shot" else FEW_SHOT_PROMPT
                    prompt = prompt_template.replace("{query}", item["query"])

                    raw, clean, ms = get_response(model, prompt, seed=seed)
                    success, reason = validate_tool_call(clean, item["id"], criteria_map)

                    # Extract category (tool name)
                    expected = item.get("expected")
                    if expected:
                        if isinstance(expected, list):
                            category = expected[0].get("name") if expected else "unknown"
                        else:
                            category = expected.get("name") if isinstance(expected, dict) else "unknown"
                    else:
                        # Fallback to criteria map
                        crit = criteria_map.get(item["id"], {})
                        category = crit.get("expected_name", "unknown")

                    log_verbose(log_f, model, item, condition, prompt, raw, clean, success, reason, ms, expected_name=category)

                    results_list.append(
                        {
                            "model": model,
                            "id": item["id"],
                            "category": category,
                            "condition": condition,
                            "success": success,
                            "reason": reason,
                            "time_ms": ms,
                            "query": item["query"],
                            "clean": clean,
                        }
                    )

                with open(RESULTS_JSON, "w") as f:
                    json.dump(results_list, f, indent=2)
                generate_markdown_report_incremental(results_list, models_to_test)
                unload_model(model)
                time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--limit", type=int, help="Limit number of prompts")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-size", type=float)
    args = parser.parse_args()

    run_experiment(
        models=args.models,
        limit=args.limit,
        fresh=args.fresh,
        seed=args.seed,
        max_size=args.max_size,
    )
