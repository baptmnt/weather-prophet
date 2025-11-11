import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTypeTemporalCorrelationCNN(nn.Module):
    def __init__(self, n_types=3, n_deltas=3, output_dim=2, hidden_dim=128, cnn_channels=32):
        super().__init__()
        self.n_types = n_types
        self.n_deltas = n_deltas
        self.cnn_channels = cnn_channels

        # --- CNN partagé pour extraire les features d'une image ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, 3, 1, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(cnn_channels, 2*cnn_channels, 3, 1, 1),
            nn.BatchNorm2d(2*cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))  # batch x channels x 1 x 1

        # --- LSTM sur les vecteurs temporels ---
        self.lstm = nn.LSTM(
            input_size=2*cnn_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # --- Fully connected ---
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        x = torch.randn(batch_size, n_types, n_deltas, H, W)
        Chaque type est séparé, et dates dans l'ordre : [h-1, j-1, j-7]
        """
        batch_size = x.size(0)
        all_seq = []

        for t in range(self.n_deltas):
            # On concatène les types pour une date t : batch x n_types x H x W
            imgs = x[:, :, t, :, :]  # batch x n_types x H x W
            # On peut traiter chaque type séparément, puis concat features
            feats = []
            for ty in range(self.n_types):
                img = imgs[:, ty, :, :].unsqueeze(1)  # batch x 1 x H x W
                f = self.cnn(img)                     # batch x channels x H' x W'
                f = self.global_pool(f)
                f = f.view(batch_size, -1)            # batch x channels
                feats.append(f)
            # concat features des types pour cette date
            feats_t = torch.cat(feats, dim=1)         # batch x (channels * n_types)
            all_seq.append(feats_t.unsqueeze(1))      # batch x 1 x (channels*n_types)

        # sequence temporelle : batch x n_deltas x (channels*n_types)
        seq = torch.cat(all_seq, dim=1)

        # passage LSTM
        lstm_out, (h_n, c_n) = self.lstm(seq)        # lstm_out: batch x n_deltas x hidden_dim
        last_h = lstm_out[:, -1, :]                  # on prend la dernière sortie temporelle

        return self.regressor(last_h)                 # batch x 2
