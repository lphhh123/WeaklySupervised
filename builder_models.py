# builder_models.py
import os

from models.PCL_OICR_model import IMU_PCL_OICR
from models.PCLHead import Head_PCL
from models.WSCNet_model import IMUWSCNet

import torch

from models.WSDDN_model import WSDDN_avg, WSDDNTransformerIMU, WSDDN
from models.WSDDN_Mamba_model import WSDDN_MambaBasic, WSDDN_MambaBi
from builder_pretrainbackbone import get_pretrained_spec, _clean_state_dict
from models.oicr_paloss_model import IMU_OICR_PALoss


def _get_feat_dim_from_pretrained_name(config) -> int:
    mcfg = config["model"]
    name = mcfg.get("pretrained_name", None)
    if name is None:
        raise KeyError("config['model']['pretrained_name'] is required (no more feat_dim/path in config).")
    spec = get_pretrained_spec(name)
    return int(spec.get("feat_dim", 512))

def build_wsddn_imu_model(config, num_classes, device):
    """
    根据 config 构建 WSDDN 系列模型（带有 Transformer / SPP / SegEnc）
    """
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "wsddn")
    # 从 pretrained_name 对应 spec 里读
    feat_dim = _get_feat_dim_from_pretrained_name(config)

    # 所有 WSDDN 家族共享的子 config
    wsddn_cfg = model_cfg.get("wsddn", {})

    if model_type == "wsddn_avg":
        model = WSDDN_avg(num_classes=num_classes, feat_dim=feat_dim)

    elif model_type == "wsddn_transformer":
        d_model = wsddn_cfg.get("d_model", feat_dim)
        nhead = wsddn_cfg.get("nhead", 4)
        num_layers = wsddn_cfg.get("num_layers", 2)
        dim_ff = wsddn_cfg.get("dim_feedforward", 4 * d_model)
        dropout = wsddn_cfg.get("dropout", 0.1)
        use_pe = wsddn_cfg.get("use_positional_encoding", True)

        model = WSDDNTransformerIMU(
            num_classes=num_classes,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            use_positional_encoding=use_pe,
        )


    elif model_type == "wsddn":
        spp_levels = tuple(wsddn_cfg.get("spp_levels", [1, 2, 4]))
        spp_pool = wsddn_cfg.get("spp_pool", "max")
        model = WSDDN(
            num_classes=num_classes,
            feat_dim=feat_dim,
            spp_levels=spp_levels,
            pool_type=spp_pool,
        )

    elif model_type == "wsddn_mamba_basic":
        spp_levels = tuple(wsddn_cfg.get("spp_levels", [1, 2, 4]))
        spp_pool = wsddn_cfg.get("spp_pool", "max")

        hidden_dim = wsddn_cfg.get("mamba_hidden_dim", feat_dim)
        m_depth = wsddn_cfg.get("mamba_depth", 2)
        m_dropout = wsddn_cfg.get("mamba_dropout", 0.1)

        model = WSDDN_MambaBasic(
            num_classes=num_classes,
            feat_dim=feat_dim,
            spp_levels=spp_levels,
            pool_type=spp_pool,
            hidden_dim=hidden_dim,
            mamba_depth=m_depth,
            mamba_dropout=m_dropout,
        )

        # ========= 双向 Mamba 版 =========
    elif model_type == "wsddn_mamba_bi":
        spp_levels = tuple(wsddn_cfg.get("spp_levels", [1, 2, 4]))
        spp_pool = wsddn_cfg.get("spp_pool", "max")

        hidden_dim = wsddn_cfg.get("mamba_hidden_dim", feat_dim)
        m_depth = wsddn_cfg.get("mamba_depth", 3)
        m_dropout = wsddn_cfg.get("mamba_dropout", 0.1)
        share_w = wsddn_cfg.get("mamba_share_weights", True)
        merge_mode = wsddn_cfg.get("mamba_merge_mode", "sum")  # "sum" / "concat"

        model = WSDDN_MambaBi(
            num_classes=num_classes,
            feat_dim=feat_dim,
            spp_levels=spp_levels,
            pool_type=spp_pool,
            hidden_dim=hidden_dim,
            mamba_depth=m_depth,
            mamba_dropout=m_dropout,
            share_weights=share_w,
            merge_mode=merge_mode,
        )

    else:
        raise ValueError(f"Unknown model type for WSDDN builder: {model_type}")

    return model.to(device)


def build_pcl_oicr_imu_model(config, num_classes, device):
    mcfg = config["model"]
    # 从 pretrained_name 对应 spec 里读
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

def build_oicrPALoss_imu_model(config, num_classes, device):
    # 从 pretrained_name 对应 spec 里读
    feat_dim = _get_feat_dim_from_pretrained_name(config)
    pcl_cfg = config["model"].get("pcl", {})
    wsddn_cfg = config["model"].get("wsddn", {})

    model = IMU_OICR_PALoss(
        feat_dim=feat_dim,
        num_classes=num_classes,
        refine_times=int(pcl_cfg.get("refine_times", 3)),
        fg_thresh=float(pcl_cfg.get("fg_thresh", 0.5)),
        bg_thresh=float(pcl_cfg.get("bg_thresh", 0.1)),
        spp_levels=wsddn_cfg.get("spp_levels", [1, 2, 4]),
        pool_type=wsddn_cfg.get("spp_pool", "max"),
        stage0_boost=float(pcl_cfg.get("stage0_boost", 3.0)),
        pa_mode=str(pcl_cfg.get("pa_mode", "sigmoid")),
        enhance_weight=bool(pcl_cfg.get("enhance_weight", False)),
    )
    return model.to(device)




def build_pclHead_model(config, num_classes, device):
    """
    创建 Head_pcl
    """
    mcfg = config["model"]["pcl"]  # 你 base_config 里就是 model.pcl

    feat_dim = int(mcfg.get("feat_dim", 512))
    refine_times = int(mcfg.get("refine_times", 3))
    use_pcl = bool(mcfg.get("use_pcl", True))

    model = Head_PCL(
        feat_dim=feat_dim,
        num_classes=num_classes,
        refine_times=refine_times,
        use_pcl=use_pcl,
        fg_thresh=float(mcfg.get("fg_thresh", 0.5)),
        bg_thresh=float(mcfg.get("bg_thresh", 0.1)),
        graph_iou_thresh=float(mcfg.get("graph_iou_thresh", 0.5)),
        max_pc_num=int(mcfg.get("max_pc_num", 3)),
        hidden_dim=int(mcfg.get("hidden_dim", 4096)),
        spp_levels=tuple(mcfg.get("spp_levels", (1, 2, 4))),
        pool_type=str(mcfg.get("pool_type", "avg")),
        adapter_cfg=mcfg.get("adapter", None),
    ).to(device)

    return model

# --------- 一个构建 WSCNet-IMU 模型的 helper（会加载预训练） ----------
def build_wscnet_imu_model(config, num_classes: int):
    """
    config["model"]["backbone_pretrained_path"]: 预训练的 CNN1DClassifier_7s backbone 权重
    config["model"]["feat_dim"]: backbone 输出通道
    """
    # 从 pretrained_name 对应 spec 里读
    feat_dim = _get_feat_dim_from_pretrained_name(config)
    num_maps = 4  # 可以写到 config 里，这里先固定
    model = IMUWSCNet(num_classes=num_classes,
                      num_maps=num_maps,
                      in_channels=30,
                      feat_dim=feat_dim)

    # 加载 7s 预训练 backbone 参数（train_pretrain_model 存的是 model.backbone.state_dict()）
    ckpt_path = config["model"]["backbone_pretrained_path"]
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.features.load_state_dict(state, strict=False)
    print("[WSCNet-IMU] load backbone from:", ckpt_path)
    print("  missing keys:", missing)
    print("  unexpected keys:", unexpected)

    return model


