import os
from pathlib import Path

SEED = 42
HIDDEN_SIZE = 64
DP_HIDDEN_SIZE = 256
K_MAX = 25

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "predictor.pt"
DIRECT_MODEL_PATH = MODELS_DIR / "direct_predictor.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
