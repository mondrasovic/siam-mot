import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers import modulated_deform_conv
from yacs.config import CfgNode


def xcorr_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depth-wise cross correlation."""
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class XCorrDepthwise(nn.Module):
    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return xcorr_depthwise(x, kernel)


class ModulatedDeformXCorrDepthwise(nn.Module):
    def __init__(self, n_channels: int, template_size: int) -> None:
        super().__init__()

        out_channels = (template_size**2) * 3
        self._conv_offset_mask = nn.Conv2d(
            n_channels, out_channels, kernel_size=3, padding=1
        )

        self._init_offset()

    def forward(
        self, sr_features: torch.Tensor, template_features: torch.Tensor
    ) -> torch.Tensor:
        offset_mask_pred = self._conv_offset_mask(sr_features)
        offset_part_1, offset_part_2, mask = torch.chunk(
            offset_mask_pred, chunks=3, dim=1
        )
        offset = torch.cat((offset_part_1, offset_part_2), dim=1)
        mask = torch.sigmoid(mask)

        batch_size, n_channels, sr_height, sr_width = sr_features.shape
        *_, t_height, t_width = template_features.shape

        sr_features = sr_features.view(1, -1, sr_height, sr_width)
        template_features = template_features.view(-1, 1, t_height, t_width)

        out = modulated_deform_conv(
            sr_features,
            offset,
            mask,
            template_features,
            None,  # bias
            1,  # stride
            0,  # padding
            1,  # dilation
            batch_size * n_channels,  # groups
            1  # deformable groups
        )
        *_, out_height, out_width = out.shape
        out = out.view(batch_size, n_channels, out_height, out_width)

        return out

    def _init_offset(self):
        self._conv_offset_mask.weight.data.zero_()
        self._conv_offset_mask.bias.data.zero_()


def build_cross_correlation_layer(cfg: CfgNode) -> nn.Module:
    if cfg.MODEL.TRACK_HEAD.USE_DEFORM_XCORR:
        xcorr = ModulatedDeformXCorrDepthwise(
            n_channels=cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS,
            template_size=cfg.MODEL.TRACK_HEAD.POOLER_RESOLUTION
        )
    else:
        xcorr = XCorrDepthwise()

    return xcorr
