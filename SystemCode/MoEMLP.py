# ✅ train_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEMLP(nn.Module):
    def __init__(self, in_features=768, hidden_features=3072, num_experts=8, top_k=2):
        super(MoEMLP, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, in_features)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        gate_logits = self.gate(x)
        top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_scores = F.softmax(top_k_values, dim=-1)
        outputs = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i].view(-1)
            expert_out = torch.stack([self.experts[idx](x[b]) for b, idx in zip(range(x.size(0)), expert_idx)])
            outputs += gate_scores[..., i, None] * expert_out
        return outputs

if __name__ == "__main__":
    print("This file defines MoEMLP only. Training logic should be placed in a different script.")
