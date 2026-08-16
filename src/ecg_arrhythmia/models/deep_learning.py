"""Deep learning models (PyTorch) for ECG arrhythmia classification."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multilayer Perceptron baseline."""

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        num_classes: int = 5
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)  # (N, 1, L) -> (N, L)
        return self.network(x)


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for ECG morphology extraction."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        fc_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 5
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        layers = []
        c_in = in_channels
        for c_out, k_size in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(c_in, c_out, kernel_size=k_size, padding=k_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            c_in = c_out

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        features = self.conv_blocks(x)
        pooled = self.global_pool(features).squeeze(-1)
        return self.fc(pooled)


class FFT2DCNN(nn.Module):
    """2D-CNN operating on STFT / Spectrogram representations of ECG beats."""

    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 16,
        conv_channels: Optional[List[int]] = None,
        dropout: float = 0.3,
        num_classes: int = 5
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        if conv_channels is None:
            conv_channels = [16, 32, 64]

        # Register Hann window buffer
        self.register_buffer("window", torch.hann_window(n_fft))

        layers = []
        c_in = 1
        for c_out in conv_channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            ])
            c_in = c_out

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)  # (N, L)
        
        # Compute STFT: (N, freq_bins, time_frames)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        spec = torch.abs(stft).unsqueeze(1)  # (N, 1, H, W)
        
        feat = self.conv_blocks(spec)
        pooled = self.global_pool(feat).flatten(1)
        return self.fc(pooled)


class BiLSTM(nn.Module):
    """Bidirectional LSTM for sequence and temporal dependency modeling."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 5
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (N, L, 1)
        elif x.ndim == 3 and x.shape[1] == 1:
            x = x.transpose(1, 2)  # (N, 1, L) -> (N, L, 1)

        out, (hn, _) = self.lstm(x)
        # Global average pooling over time
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)


class CNNTransformerHybrid(nn.Module):
    """Combines local convolutional morphology features with Transformer self-attention."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # Conv projections: (N, d_model, seq_len')
        feat = self.conv_proj(x)
        feat = feat.transpose(1, 2)  # (N, seq_len', d_model)
        trans_out = self.transformer_encoder(feat)
        pooled = torch.mean(trans_out, dim=1)
        return self.fc(pooled)


class AutoencoderClassifier(nn.Module):
    """Autoencoder feature extractor with classification head."""

    def __init__(
        self,
        input_dim: int = 200,
        latent_dim: int = 16,
        encoder_channels: Optional[List[int]] = None,
        num_classes: int = 5
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64]

        self.encoder = nn.Sequential(
            nn.Conv1d(1, encoder_channels[0], kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(encoder_channels[0], encoder_channels[1], kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[1], latent_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        latent = self.encoder(x)
        return self.classifier(latent)
