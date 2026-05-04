# Experiment 16 Results - Quantized MAS Comparison

## Overall Summary
| Model          | Accuracy (mean±std)   | Router Acc   | Specialized Agent Acc   | Avg Time   | Peak VRAM   |
|----------------|-----------------------|--------------|-------------------------|------------|-------------|
| FP32 (F32)     | 94.7% ± 0.7%          | 96.0%        | 98.6%                   | 8996ms     | 3496 MB     |
| FP16 (F16)     | 92.7% ± 0.7%          | 95.0%        | 97.5%                   | 6354ms     | 2487 MB     |
| 8-bit (Q8_0)   | 92.3% ± 0.3%          | 94.0%        | 98.2%                   | 4622ms     | 1810 MB     |
| 4-bit (Q4_K_M) | 90.3% ± 1.7%          | 94.0%        | 96.1%                   | 3363ms     | 1500 MB     |
| 4-bit (Q4_K_S) | 76.0% ± 1.3%          | 95.3%        | 79.7%                   | 3227ms     | 1449 MB     |

## Error Type by Model
| Model          |   JSON Error |   Missing Required Key |   Invalid Enum/Value |   Wrong Tool |   Other |
|----------------|--------------|------------------------|----------------------|--------------|---------|
| FP32 (F32)     |            1 |                      0 |                    3 |           12 |       0 |
| FP16 (F16)     |            1 |                      2 |                    4 |           15 |       0 |
| 8-bit (Q8_0)   |            0 |                      1 |                    4 |           18 |       0 |
| 4-bit (Q4_K_M) |            1 |                      5 |                    6 |           17 |       0 |
| 4-bit (Q4_K_S) |           16 |                     36 |                    7 |           13 |       0 |