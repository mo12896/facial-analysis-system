from pathlib import Path

# Directories
root = Path.cwd().parent

CONFIG_DIR = root / "configs"
DATA_DIR = root / "data"
LOGS_DIR = root / "logs"
DATABASE_DIR = DATA_DIR / "database"
DATABSE = DATABASE_DIR / "database.pkl"

# Files
OPENCV_MODEL = "/home/moritz/anaconda3/envs/emotion/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"


# Stdoutputs & stderrs
