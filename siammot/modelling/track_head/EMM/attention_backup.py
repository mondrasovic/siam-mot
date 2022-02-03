import abc
from typing import Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers.dcn.deform_conv_module import ModulatedDeformConvPack


class FeatureSubsetSampler(nn.Module):
    def forward(self, boxes: List[BoxList],
                targets: List[BoxList]) -> Optional[torch.Tensor]:
        return None


class PreAttentionFeatureSampler(FeatureSubsetSampler):
    def __init__(self, max_subset_size: int) -> None:
        super().__init__()

        self.max_subset_size: int = max_subset_size

    def forward(self, boxes: List[BoxList],
                targets: List[BoxList]) -> Optional[torch.Tensor]:
        pred_pos_mask = self._get_pos_mask(boxes)
        gt_pos_mask = self._get_pos_mask(targets)
        pos_mask = pred_pos_mask | gt_pos_mask
        subset_mask = pos_mask

        device = pos_mask.device

        n_pos = torch.sum(pos_mask)
        n_rem_neg = max(self.max_subset_size - n_pos, 0)

        if n_rem_neg > 0:
            neg_mask = ~pos_mask
            neg_idxs = torch.atleast_1d(torch.squeeze(neg_mask.nonzero()))
            rand_idxs = torch.randperm(len(neg_idxs), device=device)[:n_rem_neg]
            neg_mask_subset = torch.zeros_like(neg_mask, device=device)
            neg_idxs_subset = neg_idxs[rand_idxs]
            neg_mask_subset[neg_idxs_subset] = True
            subset_mask = subset_mask | neg_mask_subset

        return subset_mask

    @staticmethod
    def _get_pos_mask(boxes: List[BoxList]) -> torch.Tensor:
        pos_mask = torch.cat(
            [box.get_field('labels') for box in boxes], dtype=torch.bool
        )
        return pos_mask


class Attention(abc.ABC):
    @abc.abstractmethod
    def forward(
        self, template_features: torch.Tensor, sr_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class NoAttention(nn.Module, Attention):
    def forward(
        self, template_features: torch.Tensor, sr_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return template_features, sr_features


class SpatialAttention(nn.Module):
    def __init__(self, n_channels: int, n_query_key_channels: int) -> None:
        super().__init__()

        assert n_channels >= n_query_key_channels

        self.conv_query = self._build_conv1x1(n_channels, n_query_key_channels)
        self.conv_key = self._build_conv1x1(n_channels, n_query_key_channels)
        self.conv_value = self._build_conv1x1(n_channels, n_channels)

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = self.conv_query(features)  # [B,C',H,W]
        key = self.conv_key(features)  # [B,C',H,W]
        value = self.conv_value(features)  # [B,C,H,W]

        query = query.flatten(start_dim=2)  # [B,C',N], N = H * W
        key = key.flatten(start_dim=2)  # [B,C',N]
        value = value.flatten(start_dim=2)  # [B,C,N]

        query = torch.transpose(query, 1, 2)  # [B,N,C']
        energy = torch.bmm(query, key)  # [B,N,N]
        attention = F.softmax(energy, dim=-1)  # [B,N,N]
        attention = torch.transpose(attention, 1, 2)  # [B,N,N]

        features_flat = features.flatten(start_dim=2)  # [B,C,N]
        spatial_attention = (
            self.weight * torch.bmm(value, attention) + features_flat
        )  # [B,C,N]
        spatial_attention = spatial_attention.reshape(
            features.shape
        )  # [B,C,H,W]

        return spatial_attention

    @staticmethod
    def _build_conv1x1(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class ChannelAttentionCalc(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_flat = features.flatten(start_dim=2)  # [B,C,N], N = H * W

        query = features_flat
        key = features_flat

        key = torch.transpose(key, 1, 2)  # [B,N,C]
        energy = torch.bmm(query, key)  # [B,C,C]
        energy_new = (
            torch.max(energy, dim=-1, keepdim=True)[0].expand_as(energy) -
            energy
        )  # [B,C,C]
        channel_attention = F.softmax(energy_new, dim=-1)  # [B,C,C]

        return channel_attention


class ChannelAttentionUse(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(
        self, features: torch.Tensor, attention: torch.Tensor
    ) -> torch.Tensor:
        features_flat = features.flatten(start_dim=2)  # [B,C,N]
        value = features_flat

        channel_attention = (
            self.weight * torch.bmm(attention, value) + features_flat
        )  # [B,C,N]
        channel_attention = channel_attention.reshape(
            features.shape
        )  # [B,C,H,W]

        return channel_attention


class DeformableSiameseAttention(nn.Module, Attention):
    def __init__(self, n_channels: int, n_query_key_channels: int) -> None:
        super().__init__()

        self.template_spatial_attention = SpatialAttention(
            n_channels, n_query_key_channels
        )
        self.sr_spatial_attention = SpatialAttention(
            n_channels, n_query_key_channels
        )

        self.template_channel_attention_calc = ChannelAttentionCalc()
        self.sr_channel_attention_calc = ChannelAttentionCalc()

        self.template_channel_attention_use = ChannelAttentionUse()
        self.sr_channel_attention_use = ChannelAttentionUse()

        self.sr_to_template_cross_attention = ChannelAttentionUse()
        self.template_to_sr_cross_attention = ChannelAttentionUse()

        self.template_deform_conv = self._build_deform_conv3x3(n_channels)
        self.sr_deform_conv = self._build_deform_conv3x3(n_channels)

    def forward(
        self, template_features: torch.Tensor, sr_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        template_spatial_attention_final = self.template_spatial_attention(
            template_features
        )
        sr_spatial_attention_final = self.sr_spatial_attention(sr_features)

        template_channel_attention = self.template_channel_attention_calc(
            template_features
        )
        sr_channel_attention = self.sr_channel_attention_calc(sr_features)

        template_channel_attention_final = self.template_channel_attention_use(
            template_features, template_channel_attention
        )
        sr_channel_attention_final = self.sr_channel_attention_use(
            sr_features, sr_channel_attention
        )

        template_cross_attention_final = self.sr_to_template_cross_attention(
            template_features, sr_channel_attention
        )
        sr_cross_attention_final = self.template_to_sr_cross_attention(
            sr_features, template_channel_attention
        )

        attentional_template_features = self.template_deform_conv(
            template_spatial_attention_final +
            template_channel_attention_final + template_cross_attention_final
        )
        attentional_sr_features = self.sr_deform_conv(
            sr_spatial_attention_final + sr_channel_attention_final +
            sr_cross_attention_final
        )

        return attentional_template_features, attentional_sr_features

    @staticmethod
    def _build_deform_conv3x3(n_channels: int) -> nn.Module:
        return ModulatedDeformConvPack(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1
        )


def build_attention_subset_sampler(cfg: CfgNode) -> FeatureSubsetSampler:
    attention_subset_size = cfg.MODEL.TRACK_HEAD.ATTENTION_SUBSET_SIZE

    if attention_subset_size is None:
        attention_sampler = FeatureSubsetSampler()
    else:
        attention_sampler = PreAttentionFeatureSampler(attention_subset_size)

    return attention_sampler


def build_attention(cfg: CfgNode) -> Attention:
    if cfg.MODEL.TRACK_HEAD.USE_ATTENTION:
        attention = DeformableSiameseAttention(
            cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS,
            cfg.MODEL.TRACK_HEAD.N_QUERY_KEY_CHANNELS
        )
    else:
        attention = NoAttention()

    return attention