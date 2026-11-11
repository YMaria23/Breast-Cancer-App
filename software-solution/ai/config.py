import os

AI_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(AI_DIR)
LOG_DIR = os.path.join(SOLUTION_DIR, "logs")

AI_TIMEOUT_S = float(os.environ.get("AI_TIMEOUT_S", "120"))

HIGH_THRESHOLD = 0.10
MEDIUM_MALIGNANT_THRESHOLD = 0.03
MEDIUM_BENIGN_THRESHOLD = 0.40
