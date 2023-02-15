# import os
# import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.emotion.datahandler.dataprocessor.face_detector import create_face_detector
from src.emotion.datahandler.dataprocessor.head_pose_estimator import (
    create_head_pose_detector,
)

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)


def draw_3d_axis(ax, yaw, pitch, roll, tdx=None, tdy=None, size=100, pts68=None):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is None or tdy is None:
        tdx = 0
        tdy = 0

    minx, maxx = np.min(pts68[0][:]), np.max(pts68[0][:])
    miny, maxy = np.min(pts68[1][:]), np.max(pts68[1][:])
    llength = np.sqrt((maxx - minx) * (maxy - miny))
    size = llength * 0.5

    if pts68 is not None:
        tdx = pts68[0][30]
        tdy = pts68[1][30]

        ax.scatter(
            pts68[0][:],
            pts68[1][:],
            -1 * np.array(range(68)),
            color="black",
            s=1,
        )

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = (
        size
        * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + tdy
    )
    z1 = size * (
        -np.cos(pitch) * np.cos(roll) * np.sin(yaw) + np.sin(pitch) * np.sin(roll)
    )

    # Y-Axis pointing down. drawn in green
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = (
        size
        * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
        + tdy
    )
    z2 = size * (
        np.cos(pitch) * np.sin(yaw) * np.sin(roll) + np.cos(roll) * np.sin(pitch)
    )

    # Z-Axis pointing out of the screen. drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    z3 = -size * (np.cos(pitch) * np.cos(yaw))

    ax.plot([tdx, x1], [tdy, y1], [0, z1], "r-", linewidth=2)
    ax.plot([tdx, x2], [tdy, y2], [0, z2], "g-", linewidth=2)
    ax.plot([tdx, x3], [tdy, y3], [0, z3], "b-", linewidth=2)

    ax.set_xlim3d([200, 1600])
    ax.set_ylim3d([200, 1200])
    ax.set_zlim3d([-1000, 0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


def new_draw_3d_coor(ax, yaw, pitch, roll, tdx=None, tdy=None, size=100, pts68=None):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is None or tdy is None:
        tdx = 0
        tdy = 0

    minx, maxx = np.min(pts68[0][:]), np.max(pts68[0][:])
    miny, maxy = np.min(pts68[1][:]), np.max(pts68[1][:])
    llength = np.sqrt((maxx - minx) * (maxy - miny))
    size = llength * 0.5

    # Half of image height
    y_offset = int(1053 / 2)
    # Estimated distance in between people in the room
    tdz = -1000

    if pts68 is not None:
        # Half of image width
        if tdx < int(1848 / 2):
            tdx = pts68[0][30] + int(1848 / 2)
            tdy = pts68[1][30] + y_offset

            # Calculate the mean of the x-coordinates
            x_mean = np.mean(pts68[0])

            # Reflect the x-coordinates about the x-axis at their center point
            x_reflected = -1 * (pts68[0] - x_mean) + x_mean

            ax.scatter(
                x_reflected + int(1848 / 2),
                pts68[1][:] + y_offset,
                1 * np.array(range(68)) + tdz,
                color="black",
                s=1,
            )
        else:
            tdx = pts68[0][30] - int(1848 / 2)
            tdy = pts68[1][30] + y_offset

            # Calculate the mean of the x-coordinates
            x_mean = np.mean(pts68[0])

            # Reflect the x-coordinates about the x-axis at their center point
            x_reflected = -1 * (pts68[0] - x_mean) + x_mean

            ax.scatter(
                x_reflected - int(1848 / 2),
                pts68[1][:] + y_offset,
                1 * np.array(range(68)) + tdz,
                color="black",
                s=1,
            )

    # Define axis endpoints
    size = 100
    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw + np.pi) * np.cos(roll)) + tdx
    y1 = (
        size
        * (
            np.cos(pitch) * np.sin(roll)
            + np.cos(roll) * np.sin(pitch) * np.sin(yaw + np.pi)
        )
        + tdy
    )
    z1 = (
        size
        * (
            -np.cos(pitch) * np.cos(roll) * np.sin(yaw + np.pi)
            + np.sin(pitch) * np.sin(roll)
        )
        + tdz
    )

    # Y-Axis pointing down. drawn in green
    x2 = size * (-np.cos(yaw + np.pi) * np.sin(roll)) + tdx
    y2 = (
        size
        * (
            np.cos(pitch) * np.cos(roll)
            - np.sin(pitch) * np.sin(yaw + np.pi) * np.sin(roll)
        )
        + tdy
    )
    z2 = (
        size
        * (
            np.cos(pitch) * np.sin(yaw + np.pi) * np.sin(roll)
            + np.cos(roll) * np.sin(pitch)
        )
        + tdz
    )

    # Z-Axis pointing out of the screen. drawn in blue
    x3 = size * (np.sin(yaw + np.pi)) + tdx
    y3 = size * (-np.cos(yaw + np.pi) * np.sin(pitch)) + tdy
    z3 = -size * (np.cos(pitch) * np.cos(yaw + np.pi)) + tdz

    ax.plot([tdx, x1], [tdy, y1], [tdz, z1], "r-", linewidth=2)
    ax.plot([tdx, x2], [tdy, y2], [tdz, z2], "g-", linewidth=2)
    ax.plot([tdx, x3], [tdy, y3], [tdz, z3], "b-", linewidth=2)

    # Set axis limits and labels
    ax.set_xlim3d([200, 1600])
    ax.set_ylim3d([200, 1200])
    ax.set_zlim3d([-1000, 0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


# For testing
def draw_3d_coor(yaw, pitch, roll, tdx, tdy):
    # Define axis endpoints
    size = 100
    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = (
        size
        * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + tdy
    )
    z1 = size * (
        -np.cos(pitch) * np.cos(roll) * np.sin(yaw) + np.sin(pitch) * np.sin(roll)
    )

    # Y-Axis pointing down. drawn in green
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = (
        size
        * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
        + tdy
    )
    z2 = size * (
        np.cos(pitch) * np.sin(yaw) * np.sin(roll) + np.cos(roll) * np.sin(pitch)
    )

    # Z-Axis pointing out of the screen. drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    z3 = -size * (np.cos(pitch) * np.cos(yaw))

    ax.plot([tdx, x1], [tdy, y1], [0, z1], "r-", linewidth=2)
    ax.plot([tdx, x2], [tdy, y2], [0, z2], "g-", linewidth=2)
    ax.plot([tdx, x3], [tdy, y3], [0, z3], "b-", linewidth=2)

    # Set axis limits and labels
    ax.set_xlim3d([200, 1600])
    ax.set_ylim3d([200, 1200])
    ax.set_zlim3d([-1000, 0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=-90)

    face_detector = create_face_detector({"type": "scrfd"})
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")
    detections = face_detector.detect_faces(image)

    head_pose_detector = create_head_pose_detector({"type": "synergy"})

    detections = head_pose_detector.detect_head_pose(image, detections)
    print(image.shape)

    poses = detections.head_pose_keypoints
    for angles, translation, lmks in poses:
        if translation[1] > 600:
            draw_3d_axis(
                ax,
                angles[0],
                angles[1],
                angles[2],
                translation[0],
                translation[1],
                size=10,
                pts68=lmks,
            )
        else:
            new_draw_3d_coor(
                ax,
                angles[0],
                angles[1],
                angles[2],
                translation[0],
                translation[1],
                size=10,
                pts68=lmks,
            )

    ax.scatter(1848 / 2, (3 * 1054) / 4, -500, c="r", marker="o")
    plt.show()
