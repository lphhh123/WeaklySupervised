import torch
import sys
import os
from easydict import EasyDict as edict

          
sys.path.append(os.getcwd())
from core.model import CoLA


def check_model():
    print("Starting model structure test")

                                                
    cfg = edict()
    cfg.FEATS_DIM = 30              
    cfg.NUM_CLASSES = 30       
    cfg.R_EASY = 5
    cfg.R_HARD = 20
    cfg.m = 3
    cfg.M = 6

             
    try:
        model = CoLA(cfg).cuda()
        print("Model instantiated successfully")
        print(f"   Backbone Input Dim: {model.actionness_module.len_feature}")
    except Exception as e:
        print(f"Model instantiation failed: {e}")
        return

                                 
    # Shape: [Batch, Segments, Channels] -> [4, 2048, 30]
    dummy_input = torch.randn(4, 2048, 30).cuda()
    print(f"   Input Shape: {dummy_input.shape}")

               
    try:
        # Forward
        video_scores, contrast_pairs, actionness, cas = model(dummy_input)

              
        print("\nForward pass succeeded. Output check:")
        print(f"   - Video Scores: {video_scores.shape} (Expect: [4, 30])")
        print(f"   - Actionness:   {actionness.shape}   (Expect: [4, 2048])")
        print(f"   - CAS (Class Activation): {cas.shape} (Expect: [4, 2048, 30])")

                                  
        print("   - Contrast Pairs:")
        for key, val in contrast_pairs.items():
                                    
            print(f"     * {key}: {val.shape}")

    except RuntimeError as e:
        print(f"\nForward pass runtime error: {e}")
        print("Hint: check the permute dimension order in model.py")
    except Exception as e:
        print(f"\nUnknown error: {e}")


if __name__ == '__main__':
    check_model()
