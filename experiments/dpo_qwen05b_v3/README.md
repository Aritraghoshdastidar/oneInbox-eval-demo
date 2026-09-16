# DPO Experiment V3: dpo_qwen05b_v3

- **Paradigm**: Trajectory-Aware / Structured Tool-Use Preference Optimization
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type**: LoRA (r=16, alpha=32, dropout=0.05)
- **Target Modules**: `k_proj, q_proj, down_proj, o_proj, up_proj, v_proj, gate_proj`
- **Training Samples**: 23 trajectory pairs
- **Validation Samples**: 4 trajectory pairs (Strict Holdout: RE-004, AMB-001)
- **Epochs**: 3
- **Beta**: 0.1
- **Final Train Loss**: 0.3219451637317737
- **Final Eval Loss**: 0.08066937327384949
- **Rewards Margin**: 2.4775612354278564
- **Rewards Accuracy**: 1.0
