import torch
import torch.nn as nn
from models.CDur_model import CDur
                                               
                                                     

def count_parameters(model):
            
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
              
    input_dim = 113
    output_dim = 10               
    model = CDur(inputdim=input_dim, outputdim=output_dim)

               
    params = count_parameters(model)
    print(f"该 CDur 模型的可训练参数总量为: {params:,}")
    print("-" * 30)

                      
                                                              
    try:
        from torchsummary import summary
                                 
        summary(model, input_size=(500, input_dim), device="cpu")
    except Exception as e:
        print(f"torchsummary 运行出错（可能是维度不匹配）: {e}")
                                         
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:40} | {param.numel():,}")