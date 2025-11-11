import torch
import torch.nn as nn

class MultiChannelCNN(nn.Module):
    def __init__(self, n_types=3, n_deltas=3, output_dim=2):
        super().__init__()
        self.input_channels = n_types * n_deltas

        # --- Bloc CNN ---
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        # --- Fully connected ---
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x = torch.randn(batch_size, n_types * n_deltas, H, W)
        
        x = self.features(x)
        x = self.global_pool(x)     # batch x 256 x 1 x 1
        x = x.view(x.size(0), -1)   # batch x 256
        return self.regressor(x)    # batch x 2