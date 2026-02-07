from .XRFV2 import xrfv2_factory
                                                     

def get_backbone(dataset_name, backbone_type, in_channels, pretrained_path=None):
    """
    Unified interface: callers should only use this function.
    """
    name_lower = dataset_name.lower()

    if name_lower == 'xrfv2':
        return xrfv2_factory(backbone_type, in_channels, pretrained_path)

                      
                                                            
    elif name_lower == 'hangtime':
        return xrfv2_factory(backbone_type, in_channels, pretrained_path)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported in pre_models.")
