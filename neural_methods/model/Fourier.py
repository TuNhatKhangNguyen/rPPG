import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoToFourierModel(nn.Module):
    def __init__(self, num_frequencies=64, min_freq=0.1, max_freq=20.0):
        super(VideoToFourierModel, self).__init__()
        self.K = num_frequencies
        
        # Define the frequency bins (linearly spaced between min and max)
        # These are fixed hyperparameters, but could be made learnable parameters
        self.register_buffer('frequencies', torch.linspace(min_freq, max_freq, num_frequencies))

        # 3D Convolutional Encoder
        # Input: [N, C, T, H, W]
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # Reduce spatial resolution
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)) # Global Average Pool spatial dims
        )

        # Predict Amplitudes and Phases
        # Input to linear: [N, 32 * T]
        # Output: [N, 2 * K] (K amplitudes and K phases)
        self.fc = nn.Linear(32, 2 * num_frequencies)

    def forward(self, x):
        N, C, T, H, W = x.shape
        
        # 1. Feature Extraction
        features = self.encoder(x) # Shape: [N, 32, T, 1, 1]
        features = torch.mean(features, dim=2).squeeze(-1).squeeze(-1) # Temporal pooling to [N, 32]
        
        # 2. Parameter Prediction
        params = self.fc(features) # [N, 2 * K]
        amplitudes = torch.sigmoid(params[:, :self.K]) # Constrain amplitude > 0
        phases = torch.tanh(params[:, self.K:]) * 3.14159 # Constrain phase [-pi, pi]
        
        # 3. Fourier Reconstruction
        # Create time vector t: [T]
        t = torch.linspace(0, 1, T, device=x.device)
        
        # Expand for broadcasting: 
        # freqs: [1, K, 1], t: [1, 1, T], amps/phases: [N, K, 1]
        freqs = self.frequencies.view(1, -1, 1)
        t = t.view(1, 1, -1)
        amps = amplitudes.unsqueeze(-1)
        phs = phases.unsqueeze(-1)
        
        # Compute sinusoids: A * sin(2 * pi * f * t + phi)
        # signal_components shape: [N, K, T]
        signal_components = amps * torch.sin(2 * 3.14159 * freqs * t + phs)
        
        # Sum across frequencies to get the reconstructed signal [N, T]
        reconstructed_signal = torch.sum(signal_components, dim=1)
        
        return reconstructed_signal