import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# --- Define the model ---
class AttentionLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_len=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_len = output_len
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_len * 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        last = self.norm(attn_out[:, -1])
        pred = self.fc(last)
        return pred.view(-1, self.output_len, 2)
