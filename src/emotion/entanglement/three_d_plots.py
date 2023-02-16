# import os
# import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm

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


def plot_truncated_cone(p0, p1, R0, R1):
    """
    Based on https://stackoverflow.com/a/39823124/190597 (astrokeat)

    Args:
        p0 (_type_): Coordinates of the tip of the cone
        p1 (_type_): Coordinates of the center of the base of the cone
        R0 (_type_): Diameter of the tip of the cone
        R1 (_type_): Diameter of the base of the cone
    """
    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [
        p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i]
        for i in [0, 1, 2]
    ]
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.5)


def draw_3d_axis(
    ax, yaw, pitch, roll, tdx=None, tdy=None, size=100, pts68=None, person: str = None
):
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

        points = np.stack([pts68[0][:], pts68[1][:], -1 * np.array(range(68))], axis=1)

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
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

    size = 1100
    b_r = np.tan(np.deg2rad(30)) * size
    # Z-Axis pointing out of the screen. drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    z3 = -size * (np.cos(pitch) * np.cos(yaw))

    # Correct for 3D radial distortion ...

    ax.plot([tdx, x1], [tdy, y1], [0, z1], "r-", linewidth=2)
    ax.plot([tdx, x2], [tdy, y2], [0, z2], "g-", linewidth=2)
    ax.plot([tdx, x3], [tdy, y3], [0, z3], "b-", linewidth=2)

    p0 = np.array([tdx, tdy, 0])
    p1 = np.array([x3, y3, z3])
    plot_truncated_cone(p0, p1, 0, b_r)

    ax.text(tdx, tdy, 0, person, color="black", fontsize=12, fontweight="bold")
    ax.text(x1, y1, z1, "X", color="red")
    ax.set_xlim3d([200, 1600])
    ax.set_ylim3d([200, 1200])
    ax.set_zlim3d([-1000, 0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


def new_draw_3d_axis(
    ax, yaw, pitch, roll, tdx=None, tdy=None, size=100, pts68=None, person: str = None
):
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

            points = np.stack(
                [
                    x_reflected + int(1848 / 2),
                    pts68[1][:] + y_offset,
                    -1 * np.array(range(68)) + tdz,
                ],
                axis=1,
            )

        else:
            tdx = pts68[0][30] - int(1848 / 2)
            tdy = pts68[1][30] + y_offset

            # Calculate the mean of the x-coordinates
            x_mean = np.mean(pts68[0])

            # Reflect the x-coordinates about the x-axis at their center point
            x_reflected = -1 * (pts68[0] - x_mean) + x_mean

            points = np.stack(
                [
                    x_reflected - int(1848 / 2),
                    pts68[1][:] + y_offset,
                    1 * np.array(range(68)) + tdz,
                ],
                axis=1,
            )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
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

    size = 1100
    # Z-Axis pointing out of the screen. drawn in blue
    x3 = size * (np.sin(yaw + np.pi)) + tdx
    y3 = size * (-np.cos(yaw + np.pi) * np.sin(pitch)) + tdy
    z3 = -size * (np.cos(pitch) * np.cos(yaw + np.pi)) + tdz

    ax.plot([tdx, x1], [tdy, y1], [tdz, z1], "r-", linewidth=2)
    ax.plot([tdx, x2], [tdy, y2], [tdz, z2], "g-", linewidth=2)
    ax.plot([tdx, x3], [tdy, y3], [tdz, z3], "b-", linewidth=2)

    p0 = np.array([tdx, tdy, tdz])
    p1 = np.array([x3, y3, z3])
    plot_truncated_cone(p0, p1, 0, np.tan(np.deg2rad(30)) * size)

    # Set axis limits and labels
    ax.text(tdx, tdy, tdz, person, color="black", fontsize=12, fontweight="bold")
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
    # ax.view_init(elev=0, azim=-90)

    face_detector = create_face_detector({"type": "scrfd"})
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")
    detections = face_detector.detect_faces(image)

    head_pose_detector = create_head_pose_detector({"type": "synergy"})

    detections = head_pose_detector.detect_head_pose(image, detections)

    persons = detections.class_id
    poses = detections.head_pose_keypoints

    for person, (angles, translation, lmks) in zip(persons, poses):
        if translation[1] > int(1053 / 2):
            draw_3d_axis(
                ax,
                angles[0],
                angles[1],
                angles[2],
                translation[0],
                translation[1],
                size=10,
                pts68=lmks,
                person=person,
            )
        else:
            new_draw_3d_axis(
                ax,
                angles[0],
                angles[1],
                angles[2],
                translation[0],
                translation[1],
                size=10,
                pts68=lmks,
                person=person,
            )

    # Add camera center point
    ax.scatter(1848 / 2, (3 * 1054) / 4, -600, c="r", marker="o")
    plt.show()
