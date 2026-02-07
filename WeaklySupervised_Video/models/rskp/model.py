from types import SimpleNamespace
from RSKP_MODEL.main_branch import WSTAL


def build_args(cfg: dict) -> SimpleNamespace:
    model_cfg = cfg.get("model", {})
    rskp_cfg = model_cfg.get("rskp", {})
    return SimpleNamespace(
        w=rskp_cfg.get("w", 0.2),
        inp_feat_num=model_cfg.get("inp_feat_num", 36),
        out_feat_num=rskp_cfg.get("out_feat_num", 512),
        mu_num=rskp_cfg.get("mu_num", 8),
        em_iter=rskp_cfg.get("em_iter", 3),
        class_num=model_cfg.get("num_classes", 30),
        scale_factor=rskp_cfg.get("scale_factor", 4.0),
        dropout=rskp_cfg.get("dropout", 0.4),
    )


def build_model(cfg: dict) -> WSTAL:
    args = build_args(cfg)
    return WSTAL(args)
