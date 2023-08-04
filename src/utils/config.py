import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

SEED = 42
DATA_PATH = ROOT / "datasets"
MODELS_PATH = ROOT / "models"
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
