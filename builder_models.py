# builder_models.py

from models.PCL_OICR_model import IMU_PCL_OICR
from models.WSDDN_model import WSDDN
from builder_pretrainbackbone import get_pretrained_spec


def _get_feat_dim_from_pretrained_name(config) -> int:
    mcfg = config["model"]
    name = mcfg.get("pretrained_name", None)
    if name is None:
        raise KeyError("config['model']['pretrained_name'] is required (no more feat_dim/path in config).")
    spec = get_pretrained_spec(name)
    return int(spec.get("feat_dim", 512))

def build_wsddn_imu_model(config, num_classes, device):
    """
    根据 config 构建 WSDDN 系列模型
    """
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "wsddn")
    feat_dim = _get_feat_dim_from_pretrained_name(config)

    # 所有 WSDDN 家族共享的子 config
    wsddn_cfg = model_cfg.get("wsddn", {})

    if model_type == "wsddn":
        spp_levels = tuple(wsddn_cfg.get("spp_levels", [1, 2, 4]))
        spp_pool = wsddn_cfg.get("spp_pool", "max")
        model = WSDDN(
            num_classes=num_classes,
            feat_dim=feat_dim,
            spp_levels=spp_levels,
            pool_type=spp_pool,
        )
    else:
        raise ValueError(f"Unknown model type for WSDDN builder: {model_type}")

    return model.to(device)


def build_pcl_oicr_imu_model(config, num_classes, device):
    mcfg = config["model"]
    feat_dim = _get_feat_dim_from_pretrained_name(config)

    refine_times = mcfg.get("refine_times", 3)
    use_pcl = (mcfg.get("type", "") == "pcl_imu")

    model = IMU_PCL_OICR(
        feat_dim=feat_dim,
        num_classes=num_classes,
        refine_times=refine_times,
        use_pcl=use_pcl,
        fg_thresh=mcfg.get("fg_thresh", 0.5),
        bg_thresh=mcfg.get("bg_thresh", 0.1),
        graph_iou_thresh=mcfg.get("graph_iou_thresh", 0.5),
        max_pc_num=mcfg.get("max_pc_num", 3),
        hidden_dim=mcfg.get("hidden_dim", 4096),
    )
    return model.to(device)



