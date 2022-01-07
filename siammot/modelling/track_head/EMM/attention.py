from typing import Tuple
import torch

from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode
from maskrcnn_benchmark.layers.dcn.deform_conv_module import ModulatedDeformConvPack


class NoAttention(nn.Module):
    def forward(
        self, template_features: torch.Tensor, sr_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return template_features, sr_features


class SpatialSelfAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_query_key_channels: int,
        weight: float = 1.0
    ) -> None:
        super().__init__()

        self.weight = weight

        self.conv_query = self._build_conv1x1(n_channels, n_query_key_channels)
        self.conv_key = self._build_conv1x1(n_channels, n_query_key_channels)
        self.conv_value = self._build_conv1x1(n_channels, n_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = self.conv_query(features)  # [B,C',H,W]
        key = self.conv_key(features)  # [B,C',H,W]
        value = self.conv_value(features)  # [B,C,H,W]

        query = query.flatten(start_dim=2)  # [B,C',N], N = H * W
        key = key.flatten(start_dim=2)  # [B,C',N]
        value = value.flatten(start_dim=2)  # [B,C,N]

        query = torch.transpose(query, 1, 2)  # [B,N,C']
        attention_map = torch.bmm(query, key)  # [B,N,N]
        attention_map = F.softmax(attention_map, dim=0)  # [B,N,N]

        spatial_attention = (
            self.weight * torch.bmm(value, attention_map)
        )  # [B,C,N]
        spatial_attention = spatial_attention.reshape(
            features.shape
        )  # [B,C,H,W]

        return spatial_attention

    @staticmethod
    def _build_conv1x1(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class ChannelSelfAttention(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        self.weight = weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = features  # [B,C,H,W]
        key = features  # [B,C,H,W]
        value = features  # [B,C,H,W]

        query = query.flatten(start_dim=2)  # [B,C,N], N = H * W
        key = key.flatten(start_dim=2)  # [B,C,N]
        value = value.flatten(start_dim=2)  # [B,C,N]

        key = torch.transpose(key, 1, 2)  # [B,N,C]
        attention_map = torch.bmm(query, key)  # [B,C,C]
        attention_map = F.softmax(attention_map, dim=1)  # [B,C,C]

        channel_attention = (
            self.weight * torch.bmm(attention_map, value)
        )  # [B,C,N]
        channel_attention = channel_attention.reshape(
            features.shape
        )  # [B,C,H,W]

        return channel_attention


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_query_key_channels: int,
        spatial_attention_weight: float = 1.0,
        channel_attention_weight: float = 1.0
    ) -> None:
        super().__init__()

        self.spatial_attention = SpatialSelfAttention(
            n_channels, n_query_key_channels, spatial_attention_weight
        )
        self.channel_attention = ChannelSelfAttention(channel_attention_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_attention_bias = self.spatial_attention(features)
        channel_attention_bias = self.channel_attention(features)
        self_attention = spatial_attention_bias + channel_attention_bias

        return self_attention


class CrossAttention(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        self.weight = weight

    def forward(
        self, src_features: torch.Tensor, dst_features: torch.Tensor
    ) -> torch.Tensor:
        src_features_flat = src_features.flatten(
            start_dim=2
        )  # [B,C,N], N = H * W
        dst_features_flat = dst_features.flatten(start_dim=2)  # [B,C,N]

        src_features_t = torch.transpose(src_features_flat, 1, 2)  # [B,N,C]
        attention_map = torch.bmm(src_features_flat, src_features_t)  # [B,C,C]
        attention_map = F.softmax(attention_map, dim=1)  # [B,C,C]

        src_attention = (
            self.weight * torch.bmm(attention_map, dst_features_flat)
        )  # [B,C,N]
        src_attention = src_attention.reshape(dst_features.shape)  # [B,C,H,W]

        return src_attention


class DeformableSiameseAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_query_key_channels: int,
        spatial_attention_weight: float = 1.0,
        channel_attention_weight: float = 1.0,
        cross_attention_weight: float = 1.0
    ) -> None:
        super().__init__()

        self.template_self_attention = SelfAttention(
            n_channels, n_query_key_channels, spatial_attention_weight,
            channel_attention_weight
        )
        self.sr_self_attention = SelfAttention(
            n_channels, n_query_key_channels, spatial_attention_weight,
            channel_attention_weight
        )
        self.sr_to_template_cross_attention = CrossAttention(
            cross_attention_weight
        )
        self.template_to_sr_cross_attention = CrossAttention(
            cross_attention_weight
        )

        self.template_deform_conv = self._build_deform_conv3x3(n_channels)
        self.sr_deform_conv = self._build_deform_conv3x3(n_channels)

    def forward(
        self, template_features: torch.Tensor, sr_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        template_self_attention_bias = self.template_self_attention(
            template_features
        )
        template_cross_attention_bias = self.sr_to_template_cross_attention(
            sr_features, template_features
        )

        sr_self_attention_bias = self.sr_self_attention(sr_features)
        sr_cross_attention_bias = self.template_to_sr_cross_attention(
            template_features, sr_features
        )

        attentional_template_features = (
            template_features + template_self_attention_bias +
            template_cross_attention_bias
        )
        attentional_sr_features = (
            sr_features + sr_self_attention_bias + sr_cross_attention_bias
        )

        attentional_template_features = self.template_deform_conv(
            attentional_template_features
        )
        attentional_sr_features = self.sr_deform_conv(attentional_sr_features)

        return attentional_template_features, attentional_sr_features

    @staticmethod
    def _build_deform_conv3x3(n_channels: int) -> nn.Module:
        return ModulatedDeformConvPack(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1
        )


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


def build_attention(cfg: CfgNode) -> FeatureChannelAttention:
    if cfg.MODEL.TRACK_HEAD.USE_ATTENTION:
        n_feature_channels = cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS
        n_query_key_channels = n_feature_channels // 2  # TODO Use config.
        attention = DeformableSiameseAttention(
            n_feature_channels, n_query_key_channels
        )
    else:
        attention = NoAttention()

    return attention
