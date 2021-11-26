import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flattens each input tensor along the batch dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens the input tensor of shape [B,D1..Dn] into [B,-1].

        Args:
            x (torch.Tensor): Tensor of shape [B,D1...Dn] to flatten (reshape).

        Returns:
            torch.Tensor: Flattened tensor of shape [B,-1].
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x


class FeatureEmbHead(nn.Module):
    """Converts template (exemplar) features produced by a Siamese tracker into
    1D embedding vectors for subsequent re-identification (ReID) purposes.
    """

    def __init__(self, n_feature_channels: int, n_emb_dim: int = 256) -> None:
        """Constructor.

        Args:
            n_feature_channels (int): Number of channels in the template
            (exemplar) features.
            n_emb_dim (int, optional): Number of embedding dimensions.
            Defaults to 256.
        """
        super().__init__()

        self.conv1 = self._build_conv3x3(n_feature_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = self._build_conv3x3(n_feature_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.flatten = Flatten()

        flatten_len = n_feature_channels * 11 * 11
        hidden_layer_size = n_emb_dim * 2

        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(flatten_len, hidden_layer_size)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_layer_size, n_emb_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes embedding vectors from tracker template (exemplar) features.
        For each feature tensor in a batch, it applies nonlinear transformation
        consisting of convolutional and fully-connected layers. Afterwards,
        it L2-normalizes the vectors to project them onto a unit hypersphere.

        Args:
            features (torch.Tensor): Template features of shape [B,C,S,S], where
            C is the number of channels specified in the constructor.

        Returns:
            torch.Tensor: Embedding vectors of shape [B,D], where D is the
            embedding dimension.
        """
        x = self.conv1(features)  # [B,C,S - 2,S - 2]
        x = self.relu1(x)  # [B,C,S - 2,S - 2]
        x = self.conv2(x)  # [B,C,S - 4,S - 4]
        x = self.relu2(x)  # [B,C,S - 4,S - 4]
        x = self.flatten(x)  # [B,-1]
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # [B,D]

        # L2 normalization - project the embedding onto a unit hypersphere.
        x = F.normalize(x, dim=1)  # [B,D]

        return x
    
    @staticmethod
    def _build_conv3x3(n_channels: int) -> nn.Module:
        conv = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=3,
            bias=False
        )
        return conv
