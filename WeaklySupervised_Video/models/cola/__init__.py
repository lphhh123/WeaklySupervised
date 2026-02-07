from .model import CoLA
from .loss import TotalLoss


def build_model(cfg):
    return CoLA(cfg)

__all__ = ["CoLA", "TotalLoss", "build_model"]
