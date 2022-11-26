from enum import Enum


class FaceDetectionBackends(Enum):
    """Enum for the different DeepFace backends."""

    OPENCV = "opencv"
    DLIB = "dlib"
    SSD = "ssd"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"


class VideoCodecs(Enum):
    """Enum for the different OpenCV codecs."""

    MP4V = ".mp4"
    XVID = ".avi"


class FaceRecognitionModels(Enum):
    """Enum for the different DeepFace face recognition models."""

    VGGFACE = "VGG-Face"
    FACENET = "FaceNet"
    FACENET512 = "FaceNet512"
    OPENFACE = "OpenFace"
    DEEPFACE = "DeepFace"
    DEEPID = "DeepID"
    DLIB = "Dlib"
    ARCFACE = "ArcFace"
    SFACE = "SFace"
