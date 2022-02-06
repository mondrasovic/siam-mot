from collections import OrderedDict

import maskrcnn_benchmark.modeling.backbone.fpn as fpn_module
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from torch import nn
from yacs.config import CfgNode

from .dla import dla


@registry.BACKBONES.register("DLA-34-FPN")
@registry.BACKBONES.register("DLA-46-C-FPN")
@registry.BACKBONES.register("DLA-60-FPN")
@registry.BACKBONES.register("DLA-102-FPN")
@registry.BACKBONES.register("DLA-169-FPN")
def build_dla_fpn_backbone(cfg):
    body = dla(cfg)
    in_channels_stage2 = cfg.MODEL.DLA.DLA_STAGE2_OUT_CHANNELS
    in_channels_stage3 = cfg.MODEL.DLA.DLA_STAGE3_OUT_CHANNELS
    in_channels_stage4 = cfg.MODEL.DLA.DLA_STAGE4_OUT_CHANNELS
    in_channels_stage5 = cfg.MODEL.DLA.DLA_STAGE5_OUT_CHANNELS
    out_channels = cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS

    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2, in_channels_stage3, in_channels_stage4,
            in_channels_stage5
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in " \
        "registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    backbone = registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

    # freeze_dla_levels_if_needed(cfg, backbone)

    return backbone


def freeze_dla_levels_if_needed(cfg: CfgNode, backbone) -> None:
    n_levels_frozen = cfg.MODEL.BACKBONE.N_FIRST_LEVELS_FROZEN
    for i in range(0, n_levels_frozen):
        level_model = getattr(backbone.body, f'level{i}')
        freeze_dla_level(level_model)


def freeze_dla_level(level_model):
    for param in level_model.parameters():
        param.requires_grad = False
