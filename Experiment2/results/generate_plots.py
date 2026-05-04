import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_plots():
    json_path = os.path.join(SCRIPT_DIR, "experiment16_results.json")
    if not os.path.exists(json_path):
        print(f"Results file not found at {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Pre-process data
    models = ["FP32 (F32)", "FP16 (F16)", "8-bit (Q8_0)", "4-bit (Q4_K_M)", "4-bit (Q4_K_S)"]
    
    agg_data = []
    for m in models:
        mdf = df[df["model_label"] == m]
        if mdf.empty: continue
        
        overall_acc = mdf["success"].mean() * 100
        router_acc = mdf["router_correct"].mean() * 100
        correct_router_df = mdf[mdf["router_correct"] == 1]
        spec_acc = (correct_router_df["success"].mean() * 100) if not correct_router_df.empty else 0
        vram = mdf["vram_mb"].max()
        time_ms = mdf["time_ms"].mean()
        
        agg_data.append({
            "Model": m,
            "Overall Accuracy": overall_acc,
            "Specialized Agent Accuracy": spec_acc,
            "Router Accuracy": router_acc,
            "Peak VRAM (MB)": vram,
            "Avg Latency (ms)": time_ms
        })
        
    summary_df = pd.DataFrame(agg_data)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy Comparison Plot
    plt.figure(figsize=(10, 6))
    x = range(len(summary_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], summary_df["Overall Accuracy"], width, label='Overall Accuracy', color='#4C72B0')
    plt.bar([i + width/2 for i in x], summary_df["Specialized Agent Accuracy"], width, label='Specialized Agent Accuracy', color='#55A868')
    
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Quantization Level')
    plt.xticks(x, summary_df["Model"], rotation=15)
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "quantization_accuracy.png"), dpi=200)
    plt.close()

    # 2. VRAM and Latency Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#C44E52'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Peak VRAM (MB)', color=color)
    bars = ax1.bar(summary_df["Model"], summary_df["Peak VRAM (MB)"], color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=15)
    
    ax2 = ax1.twinx()  
    color = '#8172B3'
    ax2.set_ylabel('Avg Latency (ms)', color=color)  
    lines = ax2.plot(summary_df["Model"], summary_df["Avg Latency (ms)"], color=color, marker='o', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Memory Footprint and Inference Latency Trade-offs')
    fig.tight_layout()  
    plt.savefig(os.path.join(SCRIPT_DIR, "quantization_vram_latency.png"), dpi=200)
    plt.close()

    # 3. Error Breakdown Plot
    error_data = []
    for m in models:
        mdf = df[df["model_label"] == m]
        if mdf.empty: continue
        
        errors = mdf[mdf["success"] == 0]
        err_counts = {"Model": m, "JSON Error": 0, "Missing Key": 0, "Invalid Enum": 0, "Wrong Tool": 0}
        
        for reason in errors["reason"]:
            reason_str = str(reason).lower()
            if "json" in reason_str or "invalid_json_or_no_tool" in reason_str:
                err_counts["JSON Error"] += 1
            elif "missing_key" in reason_str:
                err_counts["Missing Key"] += 1
            elif "invalid_enum" in reason_str or "partial_match" in reason_str:
                err_counts["Invalid Enum"] += 1
            elif "wrong_tool" in reason_str:
                err_counts["Wrong Tool"] += 1
        
        error_data.append(err_counts)
        
    err_df = pd.DataFrame(error_data)
    err_df.set_index('Model').plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title('Error Type Breakdown by Quantization Level')
    plt.ylabel('Number of Errors')
    plt.xlabel('Model')
    plt.xticks(rotation=15)
    plt.legend(title='Error Type')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "quantization_errors.png"), dpi=200)
    plt.close()
    
    print("Plots generated successfully in results directory.")

if __name__ == "__main__":
    generate_plots()
