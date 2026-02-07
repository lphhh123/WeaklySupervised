import torch
import os
import torch.nn.functional as F
from .vgg1d import VGG1DBackbone
#from .tsse_models import TSSE_MambaBackbone_7s, TSSE_7s
#from .mamba1d import MambaBackbone
from .cnn1d import CNN1DBackbone


def _adapt_time_dependent_weights(state_dict, model):
    """
    [新增] 专门修复 TSSE 的尺寸不匹配问题
    通过插值将 7s 的预训练权重 (L=239/120) 拉伸到 30s 的模型 (L=1024/512)
    """
    model_dict = model.state_dict()
    adapted_dict = {}

    for k, v in state_dict.items():
        if k not in model_dict:
            continue

        target_shape = model_dict[k].shape
        source_shape = v.shape

        # 如果形状完全一样，直接用
        if target_shape == source_shape:
            adapted_dict[k] = v
            continue

        # 针对 TSSE 的特定层进行修复
        # 1. Attention w: [1, L_src, H, D] -> [1, L_tgt, H, D]
        if 'inner_attention.w' in k and len(source_shape) == 4:
            # print(f"   🔄 Resizing Attention {k}: {source_shape} -> {target_shape}")
            # Permute to [1, H*D, L] for interpolate
            L_src = source_shape[1]
            L_tgt = target_shape[1]
            H, D = source_shape[2], source_shape[3]

            # [1, L, H, D] -> [1, L, H*D] -> [1, H*D, L]
            v_perm = v.view(1, L_src, H * D).permute(0, 2, 1)

            # 插值
            v_resized = F.interpolate(v_perm, size=L_tgt, mode='linear', align_corners=False)

            # 还原: [1, H*D, L_tgt] -> [1, L_tgt, H*D] -> [1, L_tgt, H, D]
            v_final = v_resized.permute(0, 2, 1).view(1, L_tgt, H, D)
            adapted_dict[k] = v_final

        # 2. ContraNorm LayerNorm: [L_src] -> [L_tgt]
        elif ('contra_norm.layernorm.weight' in k or 'contra_norm.layernorm.bias' in k) and len(source_shape) == 1:
            # print(f"   🔄 Resizing Norm {k}: {source_shape} -> {target_shape}")
            # View as [1, 1, L]
            v_view = v.view(1, 1, -1)
            v_resized = F.interpolate(v_view, size=target_shape[0], mode='linear', align_corners=False)
            adapted_dict[k] = v_resized.view(-1)

        else:
            # 其他不匹配的情况，打印警告并跳过（或者尝试强加载看报错）
            # 通常 TSSE 只有上述两种跟时间有关
            print(f"   ⚠️ Skipping mismatched key {k}: {source_shape} vs {target_shape}")

    return adapted_dict

def xrfv2_factory(backbone_type, in_channels, pretrained_path=None):
    model = None

    if backbone_type == 'vgg':
        model = VGG1DBackbone(in_channels=in_channels)


    elif backbone_type == 'tsse_mamba':
        model = TSSE_MambaBackbone_7s(
            in_channels=in_channels,
            input_length=2048,
            embed_type="TSSE",
            tsse_layers=2,
            mamba_cfg={"layer": 4, "mamba_type": "dbm"}
        )

    elif backbone_type == 'tsse':
        model = TSSE_7s(
            in_channels=in_channels,
            input_length=2048,
            tsse_layers=2
        )


    elif backbone_type == 'mamba':
        model = MambaBackbone(
            in_channels=in_channels,  # 这里应该是 36
            d_model=256,
            n_layers=4,
            feat_dim=512,
            d_state=16,
            d_conv=4,
            expand=2
        )

    elif backbone_type == 'cnn1d':
        model = CNN1DBackbone(in_channels=in_channels)

    else:
        raise ValueError(f"Unknown XRFV2 backbone: {backbone_type}")

    # 加载权重逻辑 (保持不变)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"==> Loading XRFV2 {backbone_type} weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = {}
        for k, v in checkpoint.items():
            # 这里的 prefix 处理可能需要根据 Mamba 预训练权重的实际 key 进行微调
            # 通常也是 'backbone.' 开头
            if k.startswith('backbone.'):
                name = k.replace('backbone.', '', 1)
                state_dict[name] = v
            else:
                state_dict[k] = v

        if backbone_type in ['tsse', 'tsse_mamba']:
            print("   Feature: Adapting TSSE weights from 7s(478) to 30s(2048)...")
            state_dict = _adapt_time_dependent_weights(state_dict, model)
        model.load_state_dict(state_dict, strict=False)

    return model