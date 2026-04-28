import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///pl_predictor.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample", "pl_data.csv")

# ELO hyperparameters
ELO_K = 20
ELO_INITIAL = 1500

# Rolling form window
FORM_WINDOW = 5

# Head-to-head window
H2H_WINDOW = 5

# Model random seed for reproducibility
RANDOM_SEED = 42
