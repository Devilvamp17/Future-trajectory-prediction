import torch.nn as nn
class BetterLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_len=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_len = output_len 
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_len * 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1]
        last_hidden = self.norm(last_hidden)
        pred = self.fc(last_hidden)
        return pred.view(-1, self.output_len, 2)
