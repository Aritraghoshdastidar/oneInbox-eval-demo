# DPO Experiment V5: dpo_qwen05b_v5

**THIS IS THE FINAL TRAINING EXPERIMENT.**

- **Paradigm**: Tool-Routing Discrimination + Escalation Preservation
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type**: LoRA (r=16, alpha=32, dropout=0.05)
- **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Training Samples**: 34 trajectory pairs
- **Validation Samples**: 9 trajectory pairs (Strict Holdouts: RE-004, AMB-001, RE-005)
- **Epochs**: 3
- **Beta**: 0.1
- **Final Train Loss**: 0.334260544291249
- **Final Eval Loss**: 0.3666100800037384
- **Rewards Margin**: 1.7203533914354112
- **Rewards Accuracy**: 0.7777777777777778
