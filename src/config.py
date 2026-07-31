from pathlib import Path
import os



SEED = 42
HIDDEN_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "predictor.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)