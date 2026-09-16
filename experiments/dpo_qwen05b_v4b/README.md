# DPO Experiment V4B: dpo_qwen05b_v4b

- **Paradigm**: Targeted Escalation Preference Learning on Unknown Properties
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type**: LoRA (r=16, alpha=32, dropout=0.05)
- **Target Modules**: `v_proj, k_proj, down_proj, gate_proj, up_proj, o_proj, q_proj`
- **Training Samples**: 52 trajectory pairs
- **Validation Samples**: 6 trajectory pairs (Strict Holdouts: RE-004, AMB-001, RE-005)
- **Epochs**: 3
- **Beta**: 0.1
- **Final Train Loss**: 0.22280889503562298
- **Final Eval Loss**: 0.2517721354961395
- **Rewards Margin**: 1.6278311808904011
- **Rewards Accuracy**: 1.0
