import sys
from pathlib import Path

# Directories
root = Path.cwd()

CONFIG_DIR = root / "configs"
DATA_DIR = root / "data"
DATA_DIR_IMAGES = DATA_DIR / "images"
DATA_DIR_DATABASE = DATA_DIR / "database"
LOG_DIR = root / "logs"
IDENTITY_DIR = DATA_DIR / "identities"
MODEL_DIR = root / "models"


# Paths to external libraries
sys.path.append(f"{root}/external/bytetrack")


# Asset Files
IMAGE_PATH = DATA_DIR / "test_image.png"
VIDEO_PATH = DATA_DIR / "short_clip.mp4"
LIGHT_OPENPOSE_MODEL = MODEL_DIR / "checkpoint_iter_370000.pth"
EMOTION_ENV = Path("/home/moritz/anaconda3/envs/emotion")
OPENCV_MODEL = (
    EMOTION_ENV
    / "/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)


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

# Colors
DEFAULT_COLOR_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]
