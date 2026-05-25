#!/usr/bin/env python3
# Author: Jackson Russell
#
# handeye_solver.py
# Offline hand-eye calibration solver for the UR3e + D455 eye-in-hand system.
#
# Reads /unity/tool0_pose and /unity/apriltag_pose from one or more rosbag2
# bags, synchronises pose pairs by nearest timestamp, and solves AX = XB using
# OpenCV's calibrateHandEye.
#
# Both topics are expected in ROS FLU convention as published by
# HandEyeDataPublisher.cs in Unity.
#
# Output:
#   - T_tool0_to_camera as translation + quaternion
#   - A ready-to-paste static_transform_publisher invocation for your launch file
#
# Usage (standalone / via ros2 run):
#   python3 handeye_solver.py --bag <bag_dir> [options]
#   ros2 run ur3_perception handeye_solver --bag <bag_dir> [options]
#
# Options:
#   --bag      PATH    Path to rosbag2 directory (required; repeat for multiple bags)
#   --method   NAME    Solver: tsai | park | horaud | andreff | daniilidis (default: tsai)
#   --max-dt   MS      Max timestamp delta for sync in milliseconds (default: 50)
#   --min-rot  DEG     Skip pairs where EEF rotation change < this value (default: 3.0)
#   --tag-id   INT     AprilTag ID to filter (informational only; filtering done in Unity)

import argparse
import glob
import os
import sys

import numpy as np
import cv2

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[warn] rosbag2_py / rclpy not found - ensure ROS 2 is sourced.", file=sys.stderr)


# Bag reading 

def read_bag(bag_path: str, topics: list):
    """Read selected topics from a rosbag2 bag.

    Returns: dict[topic_name] -> list of (timestamp_ns: int, msg)
    Tries sqlite3 storage first, falls back to mcap.
    """
    if not HAS_ROS:
        raise RuntimeError("rosbag2_py is required. Source your ROS 2 workspace first.")

    for storage_id in ("sqlite3", "mcap"):
        try:
            storage_options = rosbag2_py.StorageOptions(
                uri=bag_path, storage_id=storage_id
            )
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            )
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)

            type_map = {
                info.name: info.type
                for info in reader.get_all_topics_and_types()
            }

            messages = {t: [] for t in topics}
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                if topic in topics:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    messages[topic].append((timestamp, msg))

            return messages

        except Exception:
            continue

    raise RuntimeError(
        f"Could not open bag at '{bag_path}'. "
        "Check the path and that the bag storage format is sqlite3 or mcap."
    )


# Pose conversion

def posestamped_to_matrix(msg) -> np.ndarray:
    """Convert geometry_msgs/PoseStamped to a 4x4 homogeneous transform (float64)."""
    p = msg.pose.position
    q = msg.pose.orientation  # x, y, z, w

    # Normalise quaternion
    arr = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        return np.eye(4, dtype=np.float64)
    arr /= norm
    x, y, z, w = arr

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [p.x, p.y, p.z]
    return T


# Synchronisation

def sync_pairs(tool0_msgs: list, tag_msgs: list,
               max_dt_ms: float = 50.0) -> list:
    """Match each tool0 message to the nearest-timestamp tag message.

    Returns list of (T_tool0 4x4, T_tag 4x4) pairs within max_dt_ms.
    """
    if not tag_msgs:
        return []

    max_dt_ns = int(max_dt_ms * 1e6)
    tag_times = np.array([t for t, _ in tag_msgs], dtype=np.int64)
    tag_data  = [m for _, m in tag_msgs]

    pairs = []
    for t0_time, t0_msg in tool0_msgs:
        diffs = np.abs(tag_times - t0_time)
        idx   = int(np.argmin(diffs))
        if diffs[idx] <= max_dt_ns:
            pairs.append((
                posestamped_to_matrix(t0_msg),
                posestamped_to_matrix(tag_data[idx]),
            ))
    return pairs


def filter_by_rotation(pairs: list, min_rot_deg: float = 3.0) -> list:
    """Drop consecutive pairs where the EEF rotation change is below min_rot_deg.

    Near-zero rotation is ill-conditioned for the solver. See rosbag_recording.md
    for the rotation angle formula and why this filter matters.
    """
    if not pairs:
        return pairs

    filtered = [pairs[0]]
    for i in range(1, len(pairs)):
        R_prev = filtered[-1][0][:3, :3]
        R_curr = pairs[i][0][:3, :3]
        R_rel  = R_prev.T @ R_curr
        # Clamp for numerical safety before acos
        trace  = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(trace))
        if angle_deg >= min_rot_deg:
            filtered.append(pairs[i])

    return filtered


# Solver


_METHODS = {
    "tsai":       cv2.CALIB_HAND_EYE_TSAI,
    "park":       cv2.CALIB_HAND_EYE_PARK,
    "horaud":     cv2.CALIB_HAND_EYE_HORAUD,
    "andreff":    cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve(pairs: list, method_name: str = "tsai") -> np.ndarray:
    """Run cv2.calibrateHandEye and return T_tool0_to_camera (4x4, float64).

    Solves AX = XB (eye-in-hand). See rosbag_recording.md for the full
    derivation of A, B, and the asymmetric B order that matters for residual.

    cv2.calibrateHandEye expects:
        R_gripper2base, t_gripper2base  - EEF pose in base frame   (= tool0_pose)
        R_target2cam,   t_target2cam    - tag pose in camera frame  (= apriltag_pose)
    Returns R_cam2gripper, t_cam2gripper  ->  T_tool0_to_camera (X in AX = XB).
    """
    method = _METHODS.get(method_name.lower(), cv2.CALIB_HAND_EYE_TSAI)

    R_g2b = [p[0][:3, :3]            for p in pairs]
    t_g2b = [p[0][:3, 3].reshape(3, 1) for p in pairs]
    R_t2c = [p[1][:3, :3]            for p in pairs]
    t_t2c = [p[1][:3, 3].reshape(3, 1) for p in pairs]

    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_c2g
    T[:3, 3]  = t_c2g.flatten()
    return T


def compute_residual(pairs: list, T_x: np.ndarray) -> float:
    """Mean rotation residual in degrees across all AX = XB pairs.

    See rosbag_recording.md for the residual formula and thresholds.
    Below 2 deg is good. Above 5 deg needs more rotation-diverse poses.
    """
    errors = []
    for i in range(1, len(pairs)):
        A = np.linalg.inv(pairs[i - 1][0]) @ pairs[i][0]
        B = pairs[i - 1][1] @ np.linalg.inv(pairs[i][1])
        lhs = A @ T_x
        rhs = T_x @ B
        R_err = lhs[:3, :3].T @ rhs[:3, :3]
        trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(np.degrees(np.arccos(trace)))
    return float(np.mean(errors)) if errors else float("nan")


# Output formatting

def matrix_to_quat(R: np.ndarray):
    """Rotation matrix -> quaternion (x, y, z, w) using Shepperd's method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def print_result(T: np.ndarray, n_pairs: int, residual_deg: float, method: str):
    t = T[:3, 3]
    q = matrix_to_quat(T[:3, :3])
    print()
    print("Hand-Eye Calibration Result")
    print(f"  Method      : {method}")
    print(f"  Pairs used  : {n_pairs}")
    print(f"  Residual    : {residual_deg:.3f} deg  {'(good)' if residual_deg < 2.0 else '(high - check pose diversity)'}")
    print(f"  Translation : x={t[0]: .6f}  y={t[1]: .6f}  z={t[2]: .6f}  (metres, FLU)")
    print(f"  Quaternion  : x={q[0]: .6f}  y={q[1]: .6f}  z={q[2]: .6f}  w={q[3]: .6f}")
    print()
    print("  T_tool0_to_camera (4x4, FLU convention):")
    for row in T:
        print("    " + "  ".join(f"{v:10.6f}" for v in row))
    print()
    print("Apply to Unity (HandEyeDataPublisher transform)")
    print(f"  localPosition = new Vector3({t[0]:.6f}f,  {t[1]:.6f}f,  {t[2]:.6f}f);  // FLU metres")
    print(f"  // Convert to Unity: x=-y_flu, y=z_flu, z=x_flu")
    ux, uy, uz = -t[1], t[2], t[0]
    print(f"  // Unity localPosition ~ ({ux:.6f}f, {uy:.6f}f, {uz:.6f}f)")
    print()
    print("static_transform_publisher (paste into launch file)")
    print(f"  ros2 run tf2_ros static_transform_publisher \\")
    print(f"    {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} \\")
    print(f"    {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} \\")
    print(f"    tool0 camera_optical_frame")
    print()


# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="Offline hand-eye calibration solver. Reads rosbag2, outputs T_tool0_to_camera."
    )
    parser.add_argument(
        "--bag", nargs="+", metavar="PATH",
        help="Path(s) to rosbag2 bag directory. Glob patterns are supported "
             "(e.g. /path/to/handeye_bag_*). Multiple bags are merged before solving."
    )
    parser.add_argument(
        "--bag-dir", metavar="DIR",
        help="Directory containing bag folders. All subdirectories are used as bags, "
             "sorted by name. Equivalent to --bag <DIR>/*/."
    )
    parser.add_argument(
        "--method", default="tsai",
        choices=list(_METHODS.keys()),
        help="Solver method (default: tsai)."
    )
    parser.add_argument(
        "--max-dt", type=float, default=50.0, metavar="MS",
        help="Maximum timestamp delta for pose synchronisation in ms (default: 50)."
    )
    parser.add_argument(
        "--min-rot", type=float, default=3.0, metavar="DEG",
        help="Minimum EEF rotation change between pairs in degrees (default: 3.0)."
    )
    args = parser.parse_args()

    # Resolve bag paths from --bag (with glob expansion) and/or --bag-dir
    bag_paths = []
    if args.bag:
        for pattern in args.bag:
            expanded = sorted(glob.glob(pattern))
            if expanded:
                bag_paths.extend(p for p in expanded if os.path.isdir(p))
            else:
                bag_paths.append(pattern)  # pass through; read_bag will report the error
    if args.bag_dir:
        bag_paths.extend(sorted(
            p for p in glob.glob(os.path.join(args.bag_dir, "*"))
            if os.path.isdir(p)
        ))
    if not bag_paths:
        print("[handeye_solver] ERROR: No bags specified. Use --bag or --bag-dir.",
              file=sys.stderr)
        sys.exit(1)

    TOOL0_TOPIC  = "/unity/tool0_pose"
    APRILTAG_TOPIC = "/unity/apriltag_pose"
    topics = [TOOL0_TOPIC, APRILTAG_TOPIC]

    # Accumulate messages across all provided bags
    all_tool0 = []
    all_tags  = []

    for bag_path in bag_paths:
        print(f"[handeye_solver] Reading: {bag_path}")
        msgs = read_bag(bag_path, topics)
        all_tool0.extend(msgs[TOOL0_TOPIC])
        all_tags.extend(msgs[APRILTAG_TOPIC])

    print(f"[handeye_solver] tool0 messages   : {len(all_tool0)}")
    print(f"[handeye_solver] apriltag messages : {len(all_tags)}")

    pairs = sync_pairs(all_tool0, all_tags, max_dt_ms=args.max_dt)
    print(f"[handeye_solver] Synchronised pairs: {len(pairs)}")

    pairs = filter_by_rotation(pairs, min_rot_deg=args.min_rot)
    print(f"[handeye_solver] After rotation filter (>={args.min_rot} deg): {len(pairs)}")

    if len(pairs) < 3:
        print(
            "[handeye_solver] ERROR: Need at least 3 valid pairs.\n"
            "  - Move the robot to more diverse poses (vary rotation, not just position).\n"
            "  - Check that the AprilTag is visible and --tag-id matches the physical tag.\n"
            "  - Lower --min-rot if rotation diversity is genuinely limited."
        )
        sys.exit(1)

    print(f"[handeye_solver] Solving with method: {args.method}")
    T = solve(pairs, method_name=args.method)
    residual = compute_residual(pairs, T)

    print_result(T, len(pairs), residual, args.method)


if __name__ == "__main__":
    main()
