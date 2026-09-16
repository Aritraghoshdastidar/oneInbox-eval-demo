# DPO Experiment V4A: dpo_qwen05b_v4a

- **Paradigm**: Balanced Trajectory-Aware Preference Optimization
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type**: LoRA (r=16, alpha=32, dropout=0.05)
- **Target Modules**: `up_proj, o_proj, v_proj, k_proj, gate_proj, q_proj, down_proj`
- **Training Samples**: 37 trajectory pairs
- **Validation Samples**: 4 trajectory pairs (Strict Holdout: RE-004, AMB-001)
- **Epochs**: 3
- **Beta**: 0.1
- **Final Train Loss**: 0.24635263945286473
- **Final Eval Loss**: 0.3499162793159485
- **Rewards Margin**: 0.8947778046131134
- **Rewards Accuracy**: 1.0
