import torch
import torch.nn as nn



class MultiTypeTemporalCorrelationCNN(nn.Module):
    def __init__(
        self,
        n_types: int = 3,
        n_deltas: int = 3,
        output_dim: int = 2,
        hidden_dim: int = 128,
        cnn_channels: int = 32,
        proj_dim: int | None = None  # Réduction optionnelle de l'input du LSTM
    ):
        super().__init__()
        self.n_types = n_types
        self.n_deltas = n_deltas
        self.cnn_channels = cnn_channels
        self.output_dim = output_dim

        # CNN partagé (1 canal)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(cnn_channels, 2 * cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 2*cnn_channels, 1, 1)

        # Dimensions de features
        self.per_type_feat_dim = 2 * cnn_channels
        self.concat_feat_dim = n_types * self.per_type_feat_dim  # concat des types à chaque timestep

        # Projection optionnelle avant LSTM (si proj_dim est fourni)
        if proj_dim is not None:
            self.proj = nn.Linear(self.concat_feat_dim, proj_dim)
            self.lstm_input_size = proj_dim
        else:
            self.proj = nn.Identity()
            self.lstm_input_size = self.concat_feat_dim

        # LSTM sur la séquence temporelle
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Tête de régression
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_types, n_deltas, H, W)
        """
        B, _, _, H, W = x.shape
        steps = []

        for t in range(self.n_deltas):
            # (B, n_types, H, W) -> (B*n_types, 1, H, W)
            imgs_t = x[:, :, t, :, :]
            imgs_bt = imgs_t.reshape(B * self.n_types, 1, H, W)

            # Encode CNN puis pool global -> (B*n_types, per_type_feat_dim)
            f = self.cnn(imgs_bt)
            f = self.global_pool(f).flatten(1)

            # Regrouper par batch et concaténer sur les types -> (B, concat_feat_dim)
            f = f.view(B, self.n_types, self.per_type_feat_dim)
            f = f.reshape(B, self.concat_feat_dim)

            # Projection optionnelle -> (B, lstm_input_size)
            f = self.proj(f)

            # Empiler comme pas de temps -> (B, 1, lstm_input_size)
            steps.append(f.unsqueeze(1))

        # Séquence pour LSTM -> (B, n_deltas, lstm_input_size)
        seq = torch.cat(steps, dim=1)

        lstm_out, _ = self.lstm(seq)          # (B, n_deltas, hidden_dim)
        last = lstm_out[:, -1, :]             # (B, hidden_dim)
        return self.regressor(last)           # (B, output_dim)
