from .backbones.resnet import resnet18, resnet50
from .baseline import Baseline

from .sync_bn import convert_model


def build_model(cfg):
    if cfg.MODEL.NAME == 'resnet50':
        base_builder = resnet50
    elif cfg.MODEL.NAME == 'resnet18':
        base_builder = resnet18

    base = base_builder(cfg.MODEL.LAST_STRIDE)
    base.load_param(cfg.MODEL.PRETRAIN_PATH)

    model = Baseline(cfg.DATASETS.NUM_CLASSES, base)

    return model
