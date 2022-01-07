from os import stat
import torch

from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode


class AttentionEmbMapper(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()

        self.fc1: nn.Module = nn.Linear(in_dim, hidden_dim, bias=False)
        self.relu2: nn.Module = nn.ReLU(inplace=True)
        self.fc3: nn.Module = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes attention embedding. It maps input or context features into
        an intermediate embedding space that will be used as part of the
        attention mechanism.

        Args:
            features (torch.Tensor): Feature tensor of shape [N,C], where C is
                the input dimension (no. of channels) specified in the
                constructor.
        Returns:
            torch.Tensor: Feature embeddings of shape [N,E], where E is the
                output (embedding) dimension specified in the constructor.
        """
        x = self.fc1(features)  # [N,H], where H is the hidden layer size.
        x = self.relu2(x)  # [N,H]
        x = self.fc3(x)  # [N,E]

        return x


class FeatureChannelAttention(nn.Module):
    def __init__(
        self,
        n_feature_channels: int,
        *,
        query_key_dim: int = 128,
        value_dim: int = 256,
        softmax_temperature: float = 0.01,
    ) -> None:
        super().__init__()

        self.query_mapper: nn.Module = AttentionEmbMapper(
            n_feature_channels, query_key_dim, query_key_dim
        )
        self.key_mapper: nn.Module = AttentionEmbMapper(
            n_feature_channels, query_key_dim, query_key_dim
        )
        self.value_mapper: nn.Module = AttentionEmbMapper(
            n_feature_channels, value_dim, value_dim
        )
        self.final_mapper: nn.Module = AttentionEmbMapper(
            value_dim, value_dim, n_feature_channels * 2
        )

        self.softmax_scale: float = (
            1.0 / (softmax_temperature * (n_feature_channels**0.5))
        )

    def forward(
        self, template_features: torch.Tensor, sr_features: torch.Tensor
    ) -> torch.Tensor:
        """Computes attention weight coefficients for the given template and
        search region features.

        Args:
            template_features (torch.Tensor): Template features of shape
                [N,C,T,T], where N is the batch size (no. of images) multiplied
                by the no. of proposals per image.
            sr_features (torch.Tensor): Search region features of shape
                [N,C,S,S], where N is the batch size (no. of images) multiplied
                by the no. of proposals per image, and S >= T.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of weight coefficients from
            the <0, 1> interval of shape [N,C], i.e., ([N,C], [N,C]), for both
            template and search region features.
        """
        # Apply global average pooling (GAP).
        template_features = torch.mean(template_features, dim=(2, 3))  # [N,C]
        sr_features = torch.mean(sr_features, dim=(2, 3))  # [N,C]

        queries = self.query_mapper(template_features)  # [N,D1]
        keys = self.key_mapper(sr_features)  # [N,D1]
        values = self.value_mapper(sr_features)  # [N,D2]

        queries = F.normalize(queries, dim=1)  # [N,D1]
        keys = F.normalize(keys, dim=1)  # [N,D1]

        weights = torch.matmul(queries, keys.T)  # [N,N]
        weights = F.softmax(weights * self.softmax_scale, dim=1)  # [N,N]

        values_weighted = torch.matmul(weights, values)  # [N,D2]

        attention_coefs = self.final_mapper(values_weighted)  # [N,C*2]

        template_coefs, sr_coefs = torch.split(
            attention_coefs, attention_coefs.shape[-1] // 2, dim=1
        )  # ([N,C], [N,C])

        return template_coefs, sr_coefs


def build_feature_channel_attention(cfg: CfgNode) -> FeatureChannelAttention:
    n_feature_channels = cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS
    query_key_dim = cfg.MODEL.TRACK_HEAD.ATT_QUERY_KEY_DIM
    value_dim = cfg.MODEL.TRACK_HEAD.ATT_VALUE_DIM
    softamx_temperature = cfg.MODEL.TRACK_HEAD.ATT_SOFTMAX_TEMP

    attention = FeatureChannelAttention(
        n_feature_channels,
        query_key_dim=query_key_dim,
        value_dim=value_dim,
        softmax_temperature=softamx_temperature
    )

    return attention
