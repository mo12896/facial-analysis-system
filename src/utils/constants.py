from pathlib import Path

# Directories
root = Path.cwd().parent

CONFIG_DIR = root / "configs"
DATA_DIR = root / "data"
LOGS_DIR = root / "logs"
DATABASE_DIR = DATA_DIR / "database"
MODEL_DIR = root / "models"


# Asset Files
LIGHT_OPENPOSE_MODEL = MODEL_DIR / "checkpoint_iter_370000.pth"
DATABASE = DATABASE_DIR / "database.pkl"
OPENCV_MODEL = "/home/moritz/anaconda3/envs/emotion/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"


# Stdoutputs & stderrs
