import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbHead(nn.Module):
    """Converts exemplar (template) features produced by a Siamese tracker into
    1D embedding vectors for subsequent re-identification (ReID) purposes.
    """

    def __init__(self, n_channels: int = 128) -> None:
        """Constructor.

        Args:
            n_channels (int, optional): Number of channels of template features.
            Defaults to 128.
        """
        super().__init__()

        self.conv1 = self._build_conv3x3(n_channels)
        self.conv2 = self._build_conv3x3(n_channels)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes embedding vectors from tracker template (exemplar) features.
        For each feature tensor in a batch, it applie nonlinear transformation
        followed by Global Average Pooling (GAP) along the channel dimension.
        Afterwards, it L2-normalizes the vectors to project them onto a unit
        hypersphere.

        Args:
            features (torch.Tensor): Template features of shape [B,C,S,S], where
            C is the number of channels specified in the constructor.

        Returns:
            torch.Tensor: Embedding vectors of shape [B,C].
        """
        x = self.conv1(features)  # [B, C, S - 2, S - 2]
        x = F.tanh(x)  # [B, C, S - 2, S - 2]
        x = self.conv2(x)  # [B, C, S - 4, S - 4]

        # Global Average Pooling (GAP).
        x = torch.mean(x, dim=(2, 3))  # [B, C]

        # L2 normalization - project the embedding onto a unit hypersphere.
        x = F.normalize(x, dim=1)  # [B, C]

        return x
    
    @staticmethod
    def _build_conv3x3(n_channels: int) -> nn.Module:
        conv = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=3,
            bias=False
        )
        return conv
