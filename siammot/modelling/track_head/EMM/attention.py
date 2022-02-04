import abc
from typing import Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from yacs.config import CfgNode
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers.dcn.deform_conv_module import ModulatedDeformConvPack


class AttentionProposalSampler(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, boxes: List[BoxList],
                targets: List[BoxList]) -> Optional[torch.Tensor]:
        pass


class AttentionAllProposalsSampler(AttentionProposalSampler):
    def forward(self, boxes: List[BoxList],
                targets: List[BoxList]) -> Optional[torch.Tensor]:
        return None


class AttentionHardMiningSampler(AttentionProposalSampler):
    def __init__(self):
        super().__init__()

    def forward(self, boxes: List[BoxList],
                targets: List[BoxList]) -> Optional[torch.Tensor]:
        return None


class AttentionRandomSampler(AttentionProposalSampler):
    def __init__(self, n_max_frame_samples: int) -> None:
        super().__init__()

        self.n_max_frame_samples: int = n_max_frame_samples

    def forward(self, boxes: List[BoxList],
                targets: List[BoxList]) -> Optional[torch.Tensor]:
        subset_idxs = torch.cat(
            [
                self._sample_frame_proposals_idxs(pred_boxes, gt_boxes)
                for pred_boxes, gt_boxes in zip(boxes, targets)
            ]
        )
        return subset_idxs

    def _sample_frame_proposals_idxs(
        self, pred_boxes: BoxList, gt_boxes: BoxList
    ) -> torch.Tensor:
        pred_pos_mask = pred_boxes.get_field('ids') >= 0
        gt_pos_mask = gt_boxes.get_field('ids') >= 0
        pos_mask = pred_pos_mask | gt_pos_mask
        subset_mask = pos_mask

        device = pos_mask.device

        n_pos = torch.sum(pos_mask).item()
        n_rem_neg = max(self.n_max_frame_samples - n_pos, 0)

        if n_rem_neg > 0:
            neg_mask = ~pos_mask
            neg_idxs = torch.squeeze(torch.nonzero(neg_mask), dim=1)
            rand_idxs = torch.randperm(len(neg_idxs), device=device)[:n_rem_neg]
            neg_mask_subset = torch.zeros_like(neg_mask, device=device)
            neg_idxs_subset = neg_idxs[rand_idxs]
            neg_mask_subset[neg_idxs_subset] = True
            subset_mask = subset_mask | neg_mask_subset

        subset_idxs = torch.squeeze(torch.nonzero(subset_mask), dim=1)

        return subset_idxs


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


class SpatialAttentionCalc(nn.Module):
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

        query = torch.flatten(query, start_dim=2)  # [B,C',N], N = H * W
        key = torch.flatten(key, start_dim=2)  # [B,C',N]
        value = torch.flatten(value, start_dim=2)  # [B,C,N]

        query = torch.transpose(query, 1, 2)  # [B,N,C']
        energy = torch.bmm(query, key)  # [B,N,N]
        attention = F.softmax(energy, dim=-1)  # [B,N,N]
        attention = torch.transpose(attention, 1, 2)  # [B,N,N]

        spatial_attention = (
            self.weight * torch.bmm(value, attention)
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
        features_flat = torch.flatten(
            features, start_dim=2
        )  # [B,C,N], N = H * W

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
        value = torch.flatten(features, start_dim=2)  # [B,C,N]

        if attention.ndim == 4:
            attention = torch.flatten(attention, start_dim=2)

        channel_attention = (
            self.weight * torch.bmm(attention, value)
        )  # [B,C,N]
        channel_attention = channel_attention.reshape(
            features.shape
        )  # [B,C,H,W]

        return channel_attention


class DeformableSiameseAttention(nn.Module, Attention):
    def __init__(self, n_channels: int, n_query_key_channels: int) -> None:
        super().__init__()

        self.template_spatial_attention_calc = SpatialAttentionCalc(
            n_channels, n_query_key_channels
        )
        self.sr_spatial_attention_calc = SpatialAttentionCalc(
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
        self,
        template_features: torch.Tensor,
        sr_features: torch.Tensor,
        subset_idxs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes and applies the attention biases to the given features.

        Args:
            template_features (torch.Tensor): Template features of shape
                [B,C,T,T].
            sr_features (torch.Tensor): Search region features of shape
                [B,S,S,S].
            subset_idxs (Optional[torch.Tensor], optional): Tensor of shape
                [B',] representing the indices of the subset of samples
                (features) to consider, such that B' <= B. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attentional template and search
                region features, a tuple of tensors with the shape of
                ([B,C,T,T], [B,C,S,S]).
        """
        if subset_idxs is None:
            template_features_subset = template_features  # [B,C,T,T]
            sr_features_subset = sr_features  # [B,C,S,S]
        else:
            template_features_subset = (
                template_features[subset_idxs]
            )  # [B',C,T,T]
            sr_features_subset = sr_features[subset_idxs]  # [B',C,S,S]

        template_spatial_attention_final = self.template_spatial_attention_calc(
            template_features_subset
        )  # [B|B',C,T,T]
        sr_spatial_attention_final = self.sr_spatial_attention_calc(
            sr_features_subset
        )  # [B|B',C,S,S]

        template_channel_attention = self.template_channel_attention_calc(
            template_features_subset
        )  # [B|B',C,C]
        sr_channel_attention = self.sr_channel_attention_calc(
            sr_features_subset
        )  # [B|B',C,C]

        template_channel_attention_final = self.template_channel_attention_use(
            template_features_subset, template_channel_attention
        )  # [B|B',T,T]
        sr_channel_attention_final = self.sr_channel_attention_use(
            sr_features_subset, sr_channel_attention
        )  # [B|B',S,S]

        template_cross_attention_final = self.sr_to_template_cross_attention(
            template_features_subset, sr_channel_attention
        )  # [B|B',C,T,T]
        sr_cross_attention_final = self.template_to_sr_cross_attention(
            sr_features_subset, template_channel_attention
        )  # [B|B',C,S,S]

        template_attention_combined = (
            template_spatial_attention_final +
            template_channel_attention_final + template_cross_attention_final
        )  # [B|B',C,T,T]
        sr_attention_combined = (
            sr_spatial_attention_final + sr_channel_attention_final +
            sr_cross_attention_final
        )  # [B|B',C,S,S]

        if subset_idxs is None:
            attentional_template_features = (
                template_features + template_attention_combined
            )
            attentional_sr_features = sr_features + sr_attention_combined
        else:
            attentional_template_features = torch.index_add(
                template_features,
                dim=0,
                index=subset_idxs,
                source=template_attention_combined
            )
            attentional_sr_features = torch.index_add(
                sr_features,
                dim=0,
                index=subset_idxs,
                source=sr_attention_combined
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


def build_attention_subset_sampler(cfg: CfgNode) -> AttentionProposalSampler:
    sampling_strategy = cfg.MODEL.TRACK_HEAD.ATTENTION.SAMPLING_STRATEGY
    n_max_frame_samples = cfg.MODEL.TRACK_HEAD.ATTENTION.N_MAX_FRAME_SAMPLES

    if sampling_strategy == 'all':
        attention_sampler = AttentionAllProposalsSampler()
    elif sampling_strategy == 'random':
        attention_sampler = AttentionRandomSampler(n_max_frame_samples)
    elif sampling_strategy == 'hard':
        attention_sampler = AttentionHardMiningSampler(n_max_frame_samples)
    else:
        raise ValueError("unrecognized attention proposal sampling strategy")

    return attention_sampler


def build_attention(cfg: CfgNode) -> Attention:
    if cfg.MODEL.TRACK_HEAD.ATTENTION.ENABLE:
        attention = DeformableSiameseAttention(
            cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS,
            cfg.MODEL.TRACK_HEAD.ATTENTION.N_QUERY_KEY_CHANNELS
        )
    else:
        attention = NoAttention()

    return attention
