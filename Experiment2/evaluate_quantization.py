"""
Experiment 16: Quantized MAS Comparison — Multi-Seed Benchmark
============================================================
Architecture: Router Agent -> Specialized Agents
Models: Qwen3-0.6B (FP32, FP16, Q8_0, Q4_K_M, Q4_K_S)
Dataset: 301 high-quality sequential prompts
Execution: 3 seeds (42, 123, 999)
"""

import json
import os
import re
import time
import requests
import pandas as pd
import ollama
from tabulate import tabulate

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except Exception:
    HAS_VIZ = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    {"name": "qwen3:0.6b-f32",    "label": "FP32 (F32)",     "bits": 32, "size_gb": 2.52},
    {"name": "qwen3:0.6b-fp16",   "label": "FP16 (F16)",     "bits": 16, "size_gb": 1.51},
    {"name": "qwen3:0.6b-q8_0",   "label": "8-bit (Q8_0)",   "bits": 8,  "size_gb": 0.83},
    {"name": "qwen3:0.6b",        "label": "4-bit (Q4_K_M)", "bits": 4,  "size_gb": 0.52},
    {"name": "qwen3:0.6b-q4_K_S", "label": "4-bit (Q4_K_S)", "bits": 4,  "size_gb": 0.47},
]

DATASET_FILE       = os.path.join(SCRIPT_DIR, "prompts.jsonl")
EVAL_CRITERIA_FILE = os.path.join(SCRIPT_DIR, "evaluation_criteria.jsonl")
RESULTS_DIR        = os.path.join(SCRIPT_DIR, "results")
SEEDS = [42, 999]

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────
ROUTER_PROMPT = """You are a routing classifier. Determine which category and priority a user query belongs to.

Categories:
1. IT_TICKET - Technical issues, system failures, hardware problems, outages
2. LEAVE_REQUEST - Employee leave requests, vacation, sick leave, time off
3. CUSTOMER_QUERY - Customer complaints, order issues, billing inquiries, refunds
4. KNOWLEDGE_QUERY - Policy questions, reimbursement rules, company guidelines

Respond with ONLY a JSON object in this exact format:
{"category": "CATEGORY_NAME", "priority": "PRIORITY_LEVEL", "reason": "BRIEF_REASON"}

Field rules:
- priority: MUST be one of ["Low", "Medium", "High", "Critical"]

Examples:
User: "The server is down, please fix it immediately"
{"category": "IT_TICKET", "priority": "Critical", "reason": "System-wide outage halting all work"}

User: "The conference room touch panel is unresponsive. Can someone look at it?"
{"category": "IT_TICKET", "priority": "Low", "reason": "Meeting room convenience feature, no direct impact on individual productivity or business operations"}

User: "Can you install Grammarly on my work laptop when you get a chance?"
{"category": "IT_TICKET", "priority": "Low", "reason": "Non-essential productivity add-on, no blocking issue"}

User: "I have a high fever and body ache – need leave for today and tomorrow"
{"category": "LEAVE_REQUEST", "priority": "High", "reason": "Same-day sick leave request with medical reason affecting immediate availability"}

User: "Planning a vacation next month, requesting leave from 15th to 22nd"
{"category": "LEAVE_REQUEST", "priority": "Low", "reason": "Future dated planned leave, no operational urgency"}

User: "Customer ORD9988 says the product arrived completely broken – they want a refund now"
{"category": "CUSTOMER_QUERY", "priority": "High", "reason": "Damaged goods – customer dissatisfaction high, potential refund/return needed promptly"}

User: "A client asked if we offer bulk discounts for 500 units. Please share pricing"
{"category": "CUSTOMER_QUERY", "priority": "Low", "reason": "Sales inquiry with no urgency, no immediate revenue loss or complaint"}

User: "Customer CUST5566 is threatening to cancel their annual subscription because of repeated billing errors"
{"category": "CUSTOMER_QUERY", "priority": "Critical", "reason": "Churn risk with high-value customer; immediate retention action required"}

User: "Client wants to know the status of their support ticket raised 3 days ago"
{"category": "CUSTOMER_QUERY", "priority": "Medium", "reason": "Standard follow-up, not an emergency but customer waiting for update"}

User: "What is the approval workflow for purchase requests above 2 lakh rupees?"
{"category": "KNOWLEDGE_QUERY", "priority": "Medium", "reason": "Process clarification needed to initiate a purchase, moderately urgent"}

User: "Can you share the company's policy on moonlighting?"
{"category": "KNOWLEDGE_QUERY", "priority": "Low", "reason": "General compliance inquiry, no immediate action required"}

User query: {query}
"""

IT_TICKET_PROMPT = """You are an IT Ticket Creation Agent. Your job is to extract technical issue details.
Respond with ONLY a JSON object in this format:
{"name": "create_it_ticket", "parameters": {"title": "Summary of the issue", "description": "Details of the problem", "priority": "Low/Medium/High/Critical", "department": "IT Support/Engineering/Infrastructure"}}

Rules:
1. priority: MUST be one of ["Low", "Medium", "High", "Critical"]
2. If any parameter is missing from the user query, use "" as the value.
3. title: Should be a concise summary.

Example:
User: "Printer on floor 2 is jammed"
{"name": "create_it_ticket", "parameters": {"title": "Printer Jam - Floor 2", "description": "Office printer jammed on the second floor", "priority": "Medium", "department": "IT Support"}}

User query: {query}
"""

LEAVE_REQUEST_PROMPT = """You are a Leave Request Processing Agent. Your job is to extract leave details from employee queries.
Respond with ONLY a JSON object in this format:
{"name": "process_leave_request", "parameters": { "leave_type": "Sick/Casual/Earned", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "reason": "Brief reason"}}

Rules:
1. leave_type: MUST be one of ["Sick", "Casual", "Earned", "Annual"]
2. If dates or reasons are missing, use "". Use 2026 for relative years.

Example:
User: "I need sick leave tomorrow for fever"
{"name": "process_leave_request", "parameters": { "leave_type": "Sick", "start_date": "2026-04-18", "end_date": "2026-04-18", "reason": "Fever"}}

User: "I need casual leave on 20th May for personal work."
{"name": "process_leave_request", "parameters": {"leave_type": "Casual", "start_date": "2026-05-20", "end_date": "2026-05-20", "reason": "Personal Work"}}

User query: {query}
"""

CUSTOMER_QUERY_PROMPT = """You are a Customer Query Agent. Your job is to route customer inquiries and complaints.
Respond with ONLY a JSON object in this format:
{"name": "route_customer_query", "parameters": {"query_type": "Type of inquiry", "customer_id": "CUSTxxx", "priority": "Low/Medium/High", "department": "Customer Support/Finance/Logistics/Sales"}}

Rules:
1. priority: MUST be one of ["Low", "Medium", "High"]
2. If customer_id is not mentioned, use "".

Example:
User: "Customer CUST123 wants a refund for order 999"
{"name": "route_customer_query", "parameters": {"query_type": "Refund Request", "customer_id": "CUST123", "priority": "High", "department": "Finance"}}
User: "Customer wants a to know the status for TKTID123"
{"name": "route_customer_query", "parameters": {"query_type": "Status Request", "customer_id": "", "priority": "High", "department": "Customer Support"}}

User query: {query}
"""

KNOWLEDGE_PROMPT = """You are an Internal Knowledge Agent. Your job is to extract policy lookup details.
Respond with ONLY a JSON object in this format:
{"name": "get_internal_knowledge", "parameters": {"topic": "The specific policy topic", "department": "HR/Finance/IT"}}

Example:
User: "What is the WFH policy?"
{"name": "get_internal_knowledge", "parameters": {"topic": "Work From Home Policy", "department": "HR"}}
User: "What is the policy for reimburshment of expenditure?"
{"name": "get_internal_knowledge", "parameters": {"topic": "Reimburshment Policy", "department": "Finance"}}

User query: {query}
"""

# ─────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────
def strip_think_and_fences(text: str) -> str:
    if text is None: return ""
    s = text.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def call_model(model_name: str, prompt: str, seed: int = 42, max_retries: int = 2):
    """With retry logic"""
    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            client = ollama.Client(timeout=120.0)
            messages = [
                {"role": "system", "content": "You are a specialized agent. Return ONLY valid JSON. No explanation, no markdown, no extra text."},
                {"role": "user", "content": prompt},
            ]
            
            response = client.chat(
                model=model_name,
                messages=messages,
                options={"seed": seed, "temperature": 0.1, "num_ctx": 8192}
            )
            
            raw = response["message"]["content"]
            duration_ms = (time.time() - start) * 1000
            return strip_think_and_fences(raw), duration_ms
            
        except Exception as e:
            if attempt == max_retries:
                print(f"\n[ERROR] Model {model_name} failed after {max_retries+1} attempts: {e}")
                return "ERROR", 0.0
            time.sleep(1.5)  # brief backoff
    return "ERROR", 0.0


def router_agent(model_name: str, query: str, seed: int):
    prompt = ROUTER_PROMPT.replace("{query}", query)
    raw, ms = call_model(model_name, prompt, seed)
    try:
        data = json.loads(raw)
        return data.get("category", "UNKNOWN").upper(), data.get("priority", "Medium"), raw, ms
    except Exception:
        return "UNKNOWN", "Medium", raw, ms

def specialized_agent(model_name: str, category: str, query: str, seed: int):
    if   category == "IT_TICKET":       prompt = IT_TICKET_PROMPT.replace("{query}", query)
    elif category == "LEAVE_REQUEST":   prompt = LEAVE_REQUEST_PROMPT.replace("{query}", query)
    elif category == "CUSTOMER_QUERY":  prompt = CUSTOMER_QUERY_PROMPT.replace("{query}", query)
    else:                               prompt = KNOWLEDGE_PROMPT.replace("{query}", query)
    return call_model(model_name, prompt, seed)

def unload_model(model_name: str):
    try: requests.post("http://localhost:11434/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=10)
    except Exception: pass

def load_dataset():
    with open(DATASET_FILE) as f: return [json.loads(l) for l in f if l.strip()]

def load_criteria():
    criteria = {}
    if os.path.exists(EVAL_CRITERIA_FILE):
        with open(EVAL_CRITERIA_FILE) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    criteria[c.get("prompt_id") or c.get("id")] = c
    return criteria

def _category_from_tool(tool):
    return {"create_it_ticket": "IT_TICKET", "process_leave_request": "LEAVE_REQUEST", "route_customer_query": "CUSTOMER_QUERY", "get_internal_knowledge": "KNOWLEDGE_QUERY"}.get(tool, "KNOWLEDGE_QUERY")

def validate(json_str, prompt_id, criteria_map):
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            obj = parsed[0] if parsed else {}
        else:
            obj = parsed
            
        if not isinstance(obj, dict) or not obj.get("name"):
            return 0, "invalid_json_or_no_tool"
            
        crit = criteria_map.get(prompt_id)
        if not crit:
            return 1, "ok_no_criteria"
            
        if obj.get("name") != crit.get("expected_name"):
            return 0, "wrong_tool"
            
        params = obj.get("parameters", {})
        for k in crit.get("required_keys", []):
            if k not in params or not str(params[k]).strip():
                return 0, f"missing_key:{k}"
                
        # Enum checking with leniency for priority
        global_priorities = ["low", "medium", "high", "critical"]
        for k, allowed in crit.get("strict_values", {}).items():
            val = str(params.get(k, "")).strip()
            if val.lower() not in [x.lower() for x in allowed]:
                # If it's priority, and it's a valid priority enum, allow as partial match
                if k == "priority" and val.lower() in global_priorities:
                    return 1, f"partial_match:priority={val}"
                return 0, f"invalid_enum:{k}={val}"
                
        return 1, "ok"
        
    except json.JSONDecodeError:
        return 0, "json_parse_error"
    except Exception as e:
        return 0, f"validation_error:{str(e)}"

def get_vram_mb():
    try:
        r = requests.get("http://localhost:11434/api/ps", timeout=5)
        return sum(m.get("size_vram", 0) for m in r.json().get("models", [])) / 1e6
    except Exception: return 0.0

# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────
def write_markdown_report(all_results, report_path):
    df = pd.DataFrame(all_results)
    md = "# Experiment 16 Results - Quantized MAS Comparison\n\n"
    
    summary = []
    for m in MODELS:
        mdf = df[df["model_label"] == m["label"]]
        seed_accs = [mdf[mdf["seed"] == s]["success"].mean()*100 for s in SEEDS]
        mean = sum(seed_accs)/len(seed_accs)
        std = (sum((x-mean)**2 for x in seed_accs)/len(seed_accs))**0.5
        vram = mdf["vram_mb"].max()
        summary.append([m["label"], f"{mean:.1f}% ± {std:.1f}%", f"{mdf['router_correct'].mean()*100:.1f}%", f"{mdf['time_ms'].mean():.0f}ms", f"{vram:.0f} MB"])
    
    md += "## Overall Summary\n"
    md += tabulate(summary, headers=["Model", "Accuracy (mean±std)", "Router Acc", "Avg Time", "Peak VRAM"], tablefmt="github")
    
    with open(report_path, "w") as f: f.write(md)

def save_json(results, path):
    with open(path, "w") as f: json.dump(results, f, indent=2)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data, criteria = load_dataset(), load_criteria()
    all_results = []

    for m in MODELS:
        print(f"\n▶ Model: {m['label']}")
        log_path = os.path.join(RESULTS_DIR, f"detailed_log_{m['label'].replace('/', '_').replace(' ', '_')}.txt")
        
        processed_entries = set()  # Track (seed, id) pairs
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Prompt ID:" in line and "Seed:" in line:
                        try:
                            # Parse format: Prompt ID: X | Seed: Y | ...
                            parts = line.split("|")
                            pid = int(parts[0].replace("Prompt ID:", "").strip())
                            sid = int(parts[1].replace("Seed:", "").strip())
                            processed_entries.add((sid, pid))
                        except: pass
        
        mode = "a" if processed_entries else "w"
        with open(log_path, mode, encoding="utf-8") as log_f:
            if mode == "w":
                log_f.write(f"Experiment 16 — {m['label']}\n{'='*60}\n\n")

            for seed in SEEDS:
                print(f"  Seed {seed}: ", end="", flush=True)
                pass_count = 0
                for idx, item in enumerate(data):
                    if (seed, item["id"]) in processed_entries:
                        continue
                        
                    cat, pri, r_raw, r_ms = router_agent(m["name"], item["query"], seed)
                    resp, s_ms = specialized_agent(m["name"], cat, item["query"], seed)
                    success, reason = validate(resp, item["id"], criteria)
                    
                    # Handle expected tool as list or dict
                    exp = item["expected"]
                    if isinstance(exp, list):
                        exp_name = exp[0].get("name", "") if exp else ""
                    else:
                        exp_name = exp.get("name", "") if exp else ""
                    
                    exp_cat = _category_from_tool(exp_name)
                    router_correct = 1 if cat == exp_cat else 0
                    
                    if success: pass_count += 1
                    
                    all_results.append({
                        "seed": seed, "model_label": m["label"], "id": item["id"], "query": item["query"],
                        "routed_category": cat, "expected_category": exp_cat,
                        "router_correct": router_correct, "success": success, "reason": reason,
                        "time_ms": r_ms + s_ms, "vram_mb": get_vram_mb()
                    })
                    
                    # Log detail per prompt
                    log_f.write(f"Prompt ID: {item['id']} | Seed: {seed} | Category: {cat} | Status: {success} ({reason})\n")
                    log_f.write(f"Query: {item['query']}\n")
                    log_f.write(f"Router Response: {r_raw}\n")
                    log_f.write(f"Response: {resp}\n")
                    log_f.write("-" * 50 + "\n")
                    log_f.flush()

                    # Unload model every 30 prompts to maintain clean state
                    if (idx + 1) % 30 == 0:
                        unload_model(m["name"])

                if processed_entries:
                    print(f"Resumed. Finished remaining.")
                else:
                    print(f"{pass_count}/{len(data)} pass")
        unload_model(m["name"])

    # For reporting, we might need to load previous results.json to avoid overwriting partial ones
    # but for now we'll just generate based on current all_results.
    write_markdown_report(all_results, os.path.join(RESULTS_DIR, "experiment16_results.md"))
    save_json(all_results, os.path.join(RESULTS_DIR, "experiment16_results.json"))
    print("\nBenchmark Complete.")

if __name__ == "__main__": run()
