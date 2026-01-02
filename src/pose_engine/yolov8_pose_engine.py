"""
yolov8_pose_engine.py

Script runs a pretrained YOLOv8-pose model and runs pose estimation on input frames.
Returns a dictionary mapping joint names to their (x, y, confidence) values.

Author: Natiq Ghafoor
"""

from ultralytics import YOLO  # type: ignore
import time


class PoseEngine:
    def __init__(self):
        """
        Load and initialize the pose estimation model.
        """
        # Load YOLOv8 pose model (smallest model for speed)
        self.model = YOLO("yolov8n-pose.pt")

        # Define joint names according to YOLOv8 keypoint ordering
        self.joint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def infer(self, frame):
        """
        Runs YOLOv8 pose inference on a frame and returns a dict of joints.

        Args:
            frame (np.ndarray): Input frame (BGR).

        Returns:
            Tuple[Dict[str, Tuple[float, float, float]], float]:
                - pose_dict: {joint_name: (x, y, confidence)} for each joint.
                - inference_time: model forward-pass time in seconds.
        """
        start = time.time()
        results = self.model(frame)

        # Handle case: no detections
        if (len(results) == 0 or 
            results[0].keypoints is None or 
            results[0].keypoints.data.shape[0] == 0):
            return {}, time.time() - start

        # Use first detected person only
        keypoints = results[0].keypoints.data[0].cpu().numpy()

        pose_dict = {}
        for i, name in enumerate(self.joint_names):
            x, y, conf = keypoints[i]
            pose_dict[name] = (float(x), float(y), float(conf))

        return pose_dict, time.time() - start

      
            



        