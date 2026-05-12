# Author: Jackson Russell
#
# yolo_detector.py
# ROS 2 node that runs YOLOv8 inference on the D455 colour stream and
# publishes detections for consumption by Unity (JSON string) and RViz
# (annotated image).
#
# Works with both detection models (yolov8n.pt) and segmentation models
# (yolov8n-seg.pt). Polygon contours are included in the JSON when a seg
# model is used; the field is omitted for detection-only models.
#
# Topics
#
# Subscribed:
#   /camera/camera/color/image_raw   (sensor_msgs/Image)
#
# Published:
#   /ur3/detections                  (std_msgs/String) - JSON array of dicts
#   /ur3/detections/image            (sensor_msgs/Image) - annotated BGR debug frame
#
# Parameters
#
#   image_topic         (string, default "/camera/camera/color/image_raw")
#   model_path          (string, default "yolov8n-seg.pt") - any .pt or .onnx path
#   confidence          (double, default 0.40)
#   device              (string, default "cpu") - "cpu" | "cuda:0" | "0"
#   publish_debug_image (bool,   default false)
#
# JSON format (message-level wrapper + per-detection items):
#   {
#     "k": [fx, fy, ppx, ppy],   // colour camera intrinsics (0s when not yet received)
#     "items": [
#       {
#         "label":  "cup",
#         "conf":   0.92,
#         "cx":     120.5,   // pixel coords, image frame
#         "cy":     80.1,
#         "w":      40.0,
#         "h":      55.0,
#         "x1":     100.5,   // bounding-box corners
#         "y1":     52.6,
#         "x2":     140.5,
#         "y2":     107.6,
#         "poly":   [x0,y0,x1,y1,...],                   // seg models only
#         "bbox3d": [xmin,ymin,zmin, xmax,ymax,zmax]    // metres, ROS camera frame
#                                                        // omitted when depth unavailable
#       }, ...
#     ]
#   }

import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('image_topic',         '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',          '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic',    '/camera/camera/color/camera_info')
        self.declare_parameter('model_path',          'yolov8n-seg.pt')
        self.declare_parameter('confidence',          0.40)
        self.declare_parameter('device',              'cpu')
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('detection_hz',        15.0)  # max detection publish rate
        self.declare_parameter('depth_sample_radius', 3)    # half-window for median depth

        image_topic       = self.get_parameter('image_topic').value
        depth_topic       = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        model_path        = self.get_parameter('model_path').value
        confidence        = self.get_parameter('confidence').value
        device            = self.get_parameter('device').value
        self._pub_dbg     = self.get_parameter('publish_debug_image').value
        detection_hz      = float(self.get_parameter('detection_hz').value)
        self._det_interval    = 1.0 / max(detection_hz, 0.1)
        self._last_det_time   = 0.0
        self._depth_radius    = int(self.get_parameter('depth_sample_radius').value)

        # depth + intrinsics state
        self._depth_frame = None   # latest 16UC1 numpy array (mm)
        self._fx = self._fy = self._ppx = self._ppy = None

        # Load model (auto-downloads yolov8n.pt on first run)
        self.get_logger().info(f'Loading YOLO model: {model_path}  device={device}')
        from ultralytics import YOLO  # deferred import so node starts even if ultralytics missing
        self._model      = YOLO(model_path)
        self._model_conf = float(confidence)
        self._device     = device
        self.get_logger().info('YOLO model loaded')

        self._bridge = CvBridge()

        # Use a best-effort QoS to match the camera driver's default
        _qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self._sub = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            _qos,
        )

        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._depth_callback,
            _qos,
        )

        self._info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._info_callback,
            10,
        )

        # Publishers
        self._det_pub = self.create_publisher(String, '/ur3/detections', 10)
        if self._pub_dbg:
            self._img_pub = self.create_publisher(Image, '/ur3/detections/image', 10)

        self.get_logger().info(
            f'YOLODetectorNode ready  colour={image_topic}  depth={depth_topic}'
        )

    def _depth_callback(self, msg: Image):
        try:
            self._depth_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'depth bridge error: {e}', throttle_duration_sec=5.0)

    def _info_callback(self, msg: CameraInfo):
        if self._fx is None:
            # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self._fx  = msg.k[0]
            self._fy  = msg.k[4]
            self._ppx = msg.k[2]
            self._ppy = msg.k[5]
            self.get_logger().info(
                f'Camera intrinsics received: fx={self._fx:.1f} fy={self._fy:.1f} '
                f'ppx={self._ppx:.1f} ppy={self._ppy:.1f}'
            )

    def _deproject(self, px: float, py: float) -> 'list | None':
        """Return [x, y, z] metres in camera frame, or None if depth invalid."""
        if self._depth_frame is None or self._fx is None:
            return None
        h, w = self._depth_frame.shape
        u, v = int(round(px)), int(round(py))
        r = self._depth_radius
        patch = self._depth_frame[
            max(0, v - r):min(h, v + r + 1),
            max(0, u - r):min(w, u + r + 1),
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        z_m = float(np.median(valid)) * 0.001  # mm -> m
        x_m = (px - self._ppx) * z_m / self._fx
        y_m = (py - self._ppy) * z_m / self._fy
        return [round(x_m, 4), round(y_m, 4), round(z_m, 4)]

    def _deproject_mask(self, mask_tensor, pct_lo=5, pct_hi=95) -> 'np.ndarray | None':
        """Deproject all interior pixels of a filled mask with percentile Z clipping.

        mask_tensor: ultralytics Masks.data[i] tensor (H_mask x W_mask, values 0/1)
        Returns (M, 3) 3D points in ROS camera frame, or None.

        Using interior pixels avoids the depth-edge halo where the aligned depth
        sensor bleeds background values onto foreground object boundaries.
        Percentile clipping removes the remaining outliers before computing the AABB.
        """
        if self._depth_frame is None or self._fx is None:
            return None

        dh, dw = self._depth_frame.shape

        # masks.data values are float probabilities 0..1 — threshold before cast
        mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
        mh, mw  = mask_np.shape
        if mh != dh or mw != dw:
            import cv2 as _cv2
            mask_np = _cv2.resize(mask_np, (dw, dh), interpolation=_cv2.INTER_NEAREST)

        # erode 2px inward to shed the boundary halo entirely
        import cv2 as _cv2
        kernel  = np.ones((5, 5), np.uint8)
        mask_np = _cv2.erode(mask_np, kernel, iterations=1)

        vs, us = np.where(mask_np > 0)
        if vs.size == 0:
            return None

        z_mm = self._depth_frame[vs, us].astype(np.float64)
        valid = z_mm > 0
        if not np.any(valid):
            return None

        z_v  = z_mm[valid] * 0.001   # mm -> m
        us_v = us[valid].astype(np.float64)
        vs_v = vs[valid].astype(np.float64)

        # percentile clip on Z to discard remaining boundary outliers
        z_lo = np.percentile(z_v, pct_lo)
        z_hi = np.percentile(z_v, pct_hi)
        inliers = (z_v >= z_lo) & (z_v <= z_hi)
        if not np.any(inliers):
            return None

        z  = z_v[inliers]
        pu = us_v[inliers]
        pv = vs_v[inliers]
        x  = (pu - self._ppx) * z / self._fx
        y  = (pv - self._ppy) * z / self._fy
        return np.stack([x, y, z], axis=1)

    def _image_callback(self, msg: Image):
        # drop frames that arrive before the next publish window is due
        # this keeps inference rate == detection_hz instead of camera fps,
        # freeing the GPU for Unity rendering on the same device
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_det_time < self._det_interval:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        results = self._model(
            frame,
            conf=self._model_conf,
            device=self._device,
            verbose=False,
        )[0]

        self._last_det_time = now

        detections = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            det = {
                'label': results.names[int(box.cls)],
                'conf':  round(float(box.conf), 4),
                'cx':    round((x1 + x2) / 2, 2),
                'cy':    round((y1 + y2) / 2, 2),
                'w':     round(x2 - x1, 2),
                'h':     round(y2 - y1, 2),
                'x1':   round(x1, 2),
                'y1':   round(y1, 2),
                'x2':   round(x2, 2),
                'y2':   round(y2, 2),
            }
            # polygon contour (seg models only)
            if results.masks is not None and i < len(results.masks.xy):
                pts = results.masks.xy[i]  # (N, 2) contour pixels
                det['poly'] = [round(float(v), 1) for pt in pts for v in pt]

            detections.append(det)

        out = String()
        out.data = json.dumps({'items': detections})
        self._det_pub.publish(out)

        if self._pub_dbg:
            annotated = results.plot()
            dbg_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            dbg_msg.header = msg.header
            self._img_pub.publish(dbg_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
