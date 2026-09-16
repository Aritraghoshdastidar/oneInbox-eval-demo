# DPO Experiment: dpo_qwen05b_v1

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type**: LoRA (r=8, alpha=16, dropout=0.05)
- **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Training Samples**: 15 deduplicated pairs
- **Validation Samples**: 7 deduplicated pairs
- **Epochs**: 3
- **Beta**: 0.1
- **Learning Rate**: 5e-5
- **Final Train Loss**: 0.4090
- **Final Eval Loss**: 0.3832
- **Eval Reward Margin**: +1.0610
- **Eval Accuracy**: 100.0%
