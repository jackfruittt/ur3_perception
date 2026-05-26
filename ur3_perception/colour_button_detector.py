# Author: Jackson Russell
#
# colour_button_detector.py
# ROS 2 node that detects coloured buttons in the D455 colour stream using
# HSV thresholding with optional OpenCV CUDA acceleration.
#
# BGR->HSV conversion runs on the GPU when OpenCV is built with CUDA support.
# inRange, morphological ops, and contour finding run on CPU - there is no
# GPU contour primitive in OpenCV, and the single HSV frame download is fast.
#
# Output JSON is identical in format to yolo_detector.py so Unity can
# subscribe with a second YOLODetectionSubscriber pointed at
# /ur3/button_detections with no new C# required.
#
# Topics
#
# Subscribed:
#   /camera/camera/color/image_raw                   (sensor_msgs/Image)
#   /camera/camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image)
#   /camera/camera/color/camera_info                 (sensor_msgs/CameraInfo)
#
# Published:
#   /ur3/button_detections                           (std_msgs/String) - JSON array of dicts
#   /ur3/button_detections/image                     (sensor_msgs/Image) - annotated BGR debug frame
#
# Parameters
#
#   image_topic         (string, default "/camera/camera/color/image_raw")
#   depth_topic         (string, default "/camera/camera/aligned_depth_to_color/image_raw")
#   camera_info_topic   (string, default "/camera/camera/color/camera_info")
#   buttons             (string, default see _DEFAULT_BUTTONS) - JSON array of colour definitions
#   min_area            (int,    default 200)   - minimum blob area in pixels
#   max_area            (int,    default 50000) - maximum blob area in pixels
#   detection_hz        (double, default 15.0)  - max publish rate
#   depth_sample_radius (int,    default 3)     - half-window for median depth sample
#   publish_debug_image (bool,   default false)
#   use_cuda            (bool,   default true)  - use cv2.cuda for BGR->HSV if available
#
# Button JSON format:
#   [
#     {"label": "red",   "h_lo":   0, "h_hi":  10, "s_lo": 80, "v_lo": 60},
#     {"label": "red",   "h_lo": 170, "h_hi": 180, "s_lo": 80, "v_lo": 60},
#     {"label": "green", "h_lo":  40, "h_hi":  80, "s_lo": 80, "v_lo": 60},
#     {"label": "blue",  "h_lo": 100, "h_hi": 130, "s_lo": 80, "v_lo": 60}
#   ]
#
# Red wraps around hue=0 in OpenCV HSV (0-179), so define two entries with
# the same label - they are OR-ed together before contour finding.
#
# JSON output (message-level wrapper + per-detection items):
#   {
#     "k": [fx, fy, ppx, ppy],   // colour camera intrinsics (0s when not yet received)
#     "items": [
#       {
#         "label":  "red",
#         "conf":   1.0,          // always 1.0 - threshold is deterministic
#         "cx":     120.5,        // pixel coords, image frame
#         "cy":     80.1,
#         "w":      40.0,
#         "h":      55.0,
#         "x1":     100.5,        // bounding-box corners
#         "y1":     52.6,
#         "x2":     140.5,
#         "y2":     107.6,
#         "poly":   [x0,y0,...],                         // contour pixels, flat list
#         "bbox3d": [xmin,ymin,zmin, xmax,ymax,zmax]    // metres, ROS camera frame
#                                                        // omitted when depth unavailable
#       }, ...
#     ]
#   }

import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge

_DEFAULT_BUTTONS = json.dumps([
    {"label": "red",   "h_lo":   0, "h_hi":  10, "s_lo": 80, "v_lo": 60},
    {"label": "red",   "h_lo": 170, "h_hi": 180, "s_lo": 80, "v_lo": 60},
    {"label": "green", "h_lo":  40, "h_hi":  80, "s_lo": 80, "v_lo": 60},
    {"label": "blue",  "h_lo": 100, "h_hi": 130, "s_lo": 80, "v_lo": 60},
])


def _cuda_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except AttributeError:
        return False


class ColourButtonDetectorNode(Node):
    def __init__(self):
        super().__init__('colour_button_detector')

        # Parameters
        self.declare_parameter('image_topic',         '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',         '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic',   '/camera/camera/color/camera_info')
        self.declare_parameter('buttons',             _DEFAULT_BUTTONS)
        self.declare_parameter('min_area',            200)
        self.declare_parameter('max_area',            50000)
        self.declare_parameter('detection_hz',        15.0)
        self.declare_parameter('depth_sample_radius', 3)     # half-window for median depth
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('use_cuda',            True)

        image_topic       = self.get_parameter('image_topic').value
        depth_topic       = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self._min_area     = int(self.get_parameter('min_area').value)
        self._max_area     = int(self.get_parameter('max_area').value)
        self._pub_dbg      = self.get_parameter('publish_debug_image').value
        self._depth_radius = int(self.get_parameter('depth_sample_radius').value)
        detection_hz       = float(self.get_parameter('detection_hz').value)
        self._det_interval = 1.0 / max(detection_hz, 0.1)
        self._last_det_time = 0.0
        want_cuda           = bool(self.get_parameter('use_cuda').value)

        try:
            self._buttons = json.loads(self.get_parameter('buttons').value)
        except Exception as e:
            self.get_logger().error(f'Failed to parse buttons parameter: {e}')
            self._buttons = []

        # depth + intrinsics state
        self._depth_frame = None   # latest 16UC1 numpy array (mm)
        self._fx = self._fy = self._ppx = self._ppy = None

        # CUDA: use GPU BGR->HSV when available, CPU fallback otherwise
        self._use_cuda = want_cuda and _cuda_available()
        if want_cuda and not self._use_cuda:
            self.get_logger().warn(
                'CUDA requested but cv2.cuda not available - falling back to CPU'
            )
        self.get_logger().info(
            f'ColourButtonDetector  cuda={self._use_cuda}  '
            f'buttons={[b["label"] for b in self._buttons]}'
        )

        self._bridge = CvBridge()

        # Use a best-effort QoS to match the camera driver's default
        _qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self._sub       = self.create_subscription(Image,      image_topic,       self._image_callback,  _qos)
        self._depth_sub = self.create_subscription(Image,      depth_topic,       self._depth_callback,  _qos)
        self._info_sub  = self.create_subscription(CameraInfo, camera_info_topic, self._info_callback,   10)

        # Publishers
        self._det_pub = self.create_publisher(String, '/ur3/button_detections', 10)
        if self._pub_dbg:
            self._img_pub = self.create_publisher(Image, '/ur3/button_detections/image', 10)

        self.get_logger().info(
            f'ColourButtonDetectorNode ready  colour={image_topic}  depth={depth_topic}'
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

    def _image_callback(self, msg: Image):
        # drop frames that arrive before the next publish window is due
        # this keeps detection rate == detection_hz instead of camera fps
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_det_time < self._det_interval:
            return
        self._last_det_time = now

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        detections, debug_frame = self._detect(frame)

        out      = String()
        out.data = json.dumps({
            'k':     [self._fx or 0, self._fy or 0, self._ppx or 0, self._ppy or 0],
            'items': detections,
        })
        self._det_pub.publish(out)

        if self._pub_dbg and debug_frame is not None:
            dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            dbg_msg.header = msg.header
            self._img_pub.publish(dbg_msg)

    def _detect(self, bgr: np.ndarray):
        """BGR -> HSV (GPU if available) -> per-button inRange -> contours -> 3D.

        Entries sharing a label (e.g. the two red hue ranges) are OR-ed into
        one mask before contour finding so each physical button yields one blob.
        """
        if self._use_cuda:
            gpu_bgr = cv2.cuda_GpuMat()
            gpu_bgr.upload(bgr)
            gpu_hsv = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2HSV)
            hsv = gpu_hsv.download()
        else:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        detections  = []
        debug_frame = bgr.copy() if self._pub_dbg else None

        # Merge per-label masks before contour finding
        label_masks: dict = {}
        for btn in self._buttons:
            label = btn['label']
            lo    = np.array([btn['h_lo'], btn.get('s_lo', 80), btn.get('v_lo', 60)], dtype=np.uint8)
            hi    = np.array([btn['h_hi'], 255,                 255               ], dtype=np.uint8)
            mask  = cv2.inRange(hsv, lo, hi)
            if label in label_masks:
                cv2.bitwise_or(label_masks[label], mask, dst=label_masks[label])
            else:
                label_masks[label] = mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        for label, mask in label_masks.items():
            # open removes speckle noise; close fills small holes in the blob
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self._min_area or area > self._max_area:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                cx = bx + bw / 2.0
                cy = by + bh / 2.0

                det = {
                    'label': label,
                    'conf':  1.0,
                    'cx':    round(cx, 2),
                    'cy':    round(cy, 2),
                    'w':     round(float(bw), 2),
                    'h':     round(float(bh), 2),
                    'x1':   round(float(x1), 2),
                    'y1':   round(float(y1), 2),
                    'x2':   round(float(x2), 2),
                    'y2':   round(float(y2), 2),
                    # flat [x0,y0,...] contour - same format as YOLO seg poly
                    'poly': [round(float(v), 1) for pt in cnt.reshape(-1, 2) for v in pt],
                }

                xyz = self._deproject(cx, cy)
                if xyz is not None:
                    x, y, z = xyz
                    # approximate 3D bbox: project pixel radius to metres at this depth
                    r_px = max(bw, bh) / 2.0
                    r_m  = r_px * z / (self._fx or 600.0)
                    det['bbox3d'] = [
                        round(x - r_m,  4), round(y - r_m,  4), round(z - 0.01, 4),
                        round(x + r_m,  4), round(y + r_m,  4), round(z + 0.01, 4),
                    ]

                detections.append(det)

                if debug_frame is not None:
                    cv2.drawContours(debug_frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(
                        debug_frame, label,
                        (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

        return detections, debug_frame

    def _deproject(self, px: float, py: float) -> 'list | None':
        """Return [x, y, z] metres in camera frame, or None if depth invalid."""
        if self._depth_frame is None or self._fx is None:
            return None
        h, w = self._depth_frame.shape
        u, v = int(round(px)), int(round(py))
        r     = self._depth_radius
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


def main(args=None):
    rclpy.init(args=args)
    node = ColourButtonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
