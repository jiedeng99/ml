import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim=40, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=256,
            nhead=2,
            dropout=dropout,
            batch_first=True,
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        out = self.encoder_layer(out)
        stats = out.mean(dim=1)
        return self.pred_layer(stats)
