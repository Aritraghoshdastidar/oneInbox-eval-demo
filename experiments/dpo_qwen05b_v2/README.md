# DPO Experiment: dpo_qwen05b_v2

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type**: LoRA (r=8, alpha=16, dropout=0.05)
- **Target Modules**: `q_proj, gate_proj, down_proj, k_proj, up_proj, v_proj, o_proj`
- **Training Samples**: 18 deduplicated pairs
- **Validation Samples**: 8 deduplicated pairs
- **Epochs**: 3
- **Beta**: 0.1
- **Final Train Loss**: 0.38464026848475136
- **Final Eval Loss**: 0.4514731168746948
- **Rewards Margin**: 0.6914600487798452
