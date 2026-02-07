from .XRFV2 import xrfv2_factory
# from .opportunity import opportunity_factory # 未来扩展

def get_backbone(dataset_name, backbone_type, in_channels, pretrained_path=None):
    """
    统一接口：外层只调用这个函数
    """
    name_lower = dataset_name.lower()

    if name_lower == 'xrfv2':
        return xrfv2_factory(backbone_type, in_channels, pretrained_path)

    # [新增] 支持 HANGTIME
    # HANGTIME 使用的 CNN1D 结构和 XRFV2 是一样的，可以直接复用 xrfv2_factory
    elif name_lower == 'hangtime':
        return xrfv2_factory(backbone_type, in_channels, pretrained_path)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported in pre_models.")