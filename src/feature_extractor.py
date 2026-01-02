"""
feature_extractor.py
By: Natiq Ghafoor

Extracts per-frame joint angles and angular velocities from pose keypoints.

Input format
- Each frame is a dict: joint_name -> (x, y, confidence)

Outputs
- Joint angles (radians) for a fixed set of joints (see joint_angle_map)
- Angular velocities (radians/sec), computed as frame-to-frame deltas

Notes
- If a required joint is missing in a frame, that frame is skipped for angle output.
  This means the returned angle/velocity sequences may be shorter than the original
  pose sequence.
"""

from __future__ import annotations

import numpy as np


# Defines which 3 joints form each angle measurement.
# Format: target_joint = (parent, joint, child)
# Example: right_elbow = angle at the elbow using shoulder–elbow–wrist.
joint_angle_map = {
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
}


class FeatureExtractor:
    """
    Computes joint angle and velocity features from a sequence of pose frames.

    pose_sequence:
      List[Dict[str, (x, y, conf)]]
    """

    def __init__(self, pose_sequence):
        self.pose_sequence = pose_sequence

    def compute_joint_angles(self):
        """
        Computes joint angles (radians) per frame.

        Frames missing any required joint for a given angle are skipped entirely.

        Returns:
          List[Dict[str, float]] where each dict maps joint_name -> angle_radians
        """
        angle_sequence = []

        for idx, frame in enumerate(self.pose_sequence):
            frame_angles = {}
            valid_frame = True

            for joint_name, (A_name, B_name, C_name) in joint_angle_map.items():
                # Require all 3 joints to compute this angle.
                if A_name not in frame or B_name not in frame or C_name not in frame:
                    valid_frame = False
                    print(f"[FeatureExtractor] Skipped frame {idx}: missing {A_name}/{B_name}/{C_name}")
                    break

                A = np.array(frame[A_name][:2])
                B = np.array(frame[B_name][:2])
                C = np.array(frame[C_name][:2])

                # Angle at B between vectors BA and BC.
                v1 = A - B
                v2 = C - B

                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)

                # Degenerate vectors yield an unstable angle; return 0.0 in that case.
                if norm_v1 == 0 or norm_v2 == 0:
                    angle_value = 0.0
                else:
                    angle_value = np.arccos(dot_product / (norm_v1 * norm_v2))

                frame_angles[joint_name] = angle_value

            if valid_frame:
                angle_sequence.append(frame_angles)

        return angle_sequence

    def compute_velocity(self, fps=30):
        """
        Computes angular velocity (radians/sec) as delta(angle) per frame.

        Args:
          fps: Frames per second used to convert per-frame deltas into per-second.

        Returns:
          List[Dict[str, float]] where each dict maps joint_name -> velocity_rad_per_sec
        """
        angle_sequence = self.compute_joint_angles()
        delta_t = 1 / fps
        velocity_sequence = []

        # Velocity is defined between consecutive angle frames.
        for i in range(1, len(angle_sequence)):
            frame_velocity = {}
            for joint_name in joint_angle_map:
                theta_curr = angle_sequence[i][joint_name]
                theta_prev = angle_sequence[i - 1][joint_name]
                frame_velocity[joint_name] = (theta_curr - theta_prev) / delta_t
            velocity_sequence.append(frame_velocity)

        return velocity_sequence

    def get_features(self):
        """
        Combines angles and velocities into a single per-frame feature dict.

        Returns:
          List[Dict[str, float]] with keys:
            <joint>_angle, <joint>_velocity
        """
        angle_sequence = self.compute_joint_angles()
        velocity_sequence = self.compute_velocity()
        features = []

        # velocity_sequence is one shorter than angle_sequence.
        for i in range(1, len(angle_sequence)):
            frame_feature = {}
            for joint_name in joint_angle_map:
                frame_feature[f"{joint_name}_angle"] = angle_sequence[i][joint_name]
                frame_feature[f"{joint_name}_velocity"] = velocity_sequence[i - 1][joint_name]
            features.append(frame_feature)

        return features
