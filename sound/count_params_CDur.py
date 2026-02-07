import torch
import torch.nn as nn
from models.CDur_model import CDur
# 如果你的类定义在 model.py 中，就用 from model import CDur
# 如果是新建文件，请把之前的 CDur, Block1d, init_weights 等代码全部贴在上面

def count_parameters(model):
    # 统计总参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 1. 实例化模型
    input_dim = 113
    output_dim = 10  # 根据你的实际类别数修改
    model = CDur(inputdim=input_dim, outputdim=output_dim)

    # 2. 打印基础统计
    params = count_parameters(model)
    print(f"该 CDur 模型的可训练参数总量为: {params:,}")
    print("-" * 30)

    # 3. 进阶：查看每一层的具体分布
    # 注意：CDur 的 input_size 是 (Time, Dim)，对应 forward 里的 x.shape
    try:
        from torchsummary import summary
        # 这里模拟 500 个时间步，113 个特征维度
        summary(model, input_size=(500, input_dim), device="cpu")
    except Exception as e:
        print(f"torchsummary 运行出错（可能是维度不匹配）: {e}")
        # 如果 torchsummary 出错，可以用这个简单循环代替：
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:40} | {param.numel():,}")