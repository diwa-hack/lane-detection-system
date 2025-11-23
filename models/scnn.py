import torch
import torch.nn as nn

class SCNN(nn.Module):
    """
    Enhanced SCNN with deeper architecture (from your training notebook)
    Architecture: 64-128-256-512-1024 encoder, 1024-512-256-128-64 decoder
    """
    def __init__(self, input_channels=3, num_classes=1):
        super(SCNN, self).__init__()

        # Encoder blocks: 64-128-256-512-1024
        self.encoder = nn.ModuleList([
            # Block 1: 3 -> 64
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 2: 64 -> 128
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 3: 128 -> 256
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 4: 256 -> 512
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            # Block 5: 512 -> 1024
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
        ])

        # Decoder blocks: 1024-512-256-128-64
        self.decoder = nn.ModuleList([
            # 1024 -> 512
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # 512 -> 256
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # 256 -> 128
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # 128 -> 64
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])

        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        for layer in self.encoder:
            x = layer(x)

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        # Final output
        x = self.final(x)
        return x
