import torch
from torch import nn


class ImportanceLSTM(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=384, num_layers=1, dropout=0.2, num_feats=0):
        super().__init__()
        self.num_feats = num_feats
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # The input to the MLP is the LSTM's last hidden state + the target embedding (+ target RF features)
        mlp_input_dim = hidden_dim + embed_dim + num_feats

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_input_dim // 2, 1)
        )

    def forward(self, context, target, target_feats=None):
        # context: (batch_size, window_size, embed_dim)
        # target: (batch_size, embed_dim)
        # target_feats: (batch_size, num_feats), only fed to the MLP head
        if self.num_feats > 0:
            if target_feats is None:
                raise ValueError(f"Model was built with num_feats={self.num_feats} but no target_feats were passed")
            target = torch.cat((target, target_feats), dim=1)

        # Pass context through LSTM
        _, (hn, _) = self.lstm(context)

        # Get the final hidden state (from the last layer)
        # hn shape: (num_layers, batch_size, hidden_dim)
        final_hidden = hn[-1] # (batch_size, hidden_dim)

        # Concatenate final hidden state with target embedding
        combined = torch.cat((final_hidden, target), dim=1)

        # Pass through MLP to get logits
        logits = self.mlp(combined) # (batch_size, 1)
        return logits.squeeze(1)
