from pathlib import Path

# Directories
root = Path.cwd().parent

CONFIG_DIR = root / "configs"
DATA_DIR = root / "data"
LOGS_DIR = root / "logs"
DATABASE_DIR = DATA_DIR / "database"
IDENTITY_DIR = DATA_DIR / "identity"
MODEL_DIR = root / "models"


# Asset Files
LIGHT_OPENPOSE_MODEL = MODEL_DIR / "checkpoint_iter_370000.pth"
DATABASE = DATABASE_DIR / "database.pkl"
OPENCV_MODEL = "/home/moritz/anaconda3/envs/emotion/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"


# EmotionSets
THREE_EMOTIONS = ["happy", "neutral", "sad"]
EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "enjoyment", "fear", "sadness"]
NEUTRAL_EKMAN_EMOTIONS = [
    "anger",
    "surprise",
    "disgust",
    "enjoyment",
    "fear",
    "sadness",
    "neutral",
]
HUME_AI_EMOTIONS = [
    "admiration",
    "adoration",
    "aesthetic appreciation",
    "amusement",
    "anger",
    "anxiety",
    "awe",
    "awkwardness",
    "neutral",
    "bordeom",
    "calmness",
    "confusion",
    "contempt",
    "craving",
    "disappointment",
    "disgust",
    "admiration",
    "adoration",
    "empathetic pain",
    "entrancement",
    "envy",
    "excitement",
    "fear",
    "guilt",
    "horror",
    "interest",
    "joy",
    "nostalgia",
    "pride",
    "relief",
    "romance",
    "sadness",
    "satisfaction",
    "secual desire",
    "surprise",
    "sympathy",
    "triumph",
]

# Keypoint Definitions

# Stdoutputs & stderrs
