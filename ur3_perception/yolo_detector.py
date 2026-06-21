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
#   imgsz               (int,    default 480) - inference resolution (longest side)
#   use_openvino        (bool,   default true) - CPU only; export+load OpenVINO IR
#   torch_threads       (int,    default 0) - cap CPU threads (0 = torch default)
#
# JSON format (message-level wrapper + per-detection items):
#   {
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
#         "poly":   [x0,y0,x1,y1,...]                    // seg models only
#       }, ...
#     ]
#   }

import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('image_topic',         '/camera/camera/color/image_raw')
        self.declare_parameter('model_path',          'yolov8n-seg.pt')
        self.declare_parameter('confidence',          0.40)
        self.declare_parameter('device',              'cpu')
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('detection_hz',        15.0)  # max detection publish rate
        self.declare_parameter('max_detections',      20)   # cap detections per frame
        self.declare_parameter('max_poly_points',     50)   # stride-subsample polygons to this; 0 = skip polygons
        self.declare_parameter('use_half',            True) # FP16 on CUDA; ignored on CPU
        self.declare_parameter('imgsz',               480)  # inference resolution (longest side); lower = faster on CPU
        self.declare_parameter('torch_threads',       0)    # cap CPU threads (0 = leave torch default)
        self.declare_parameter('use_openvino',        True) # CPU: export+load OpenVINO IR (~2-3x on Intel); ignored on CUDA

        image_topic       = self.get_parameter('image_topic').value
        model_path        = self.get_parameter('model_path').value
        confidence        = self.get_parameter('confidence').value
        device            = self.get_parameter('device').value
        self._pub_dbg     = self.get_parameter('publish_debug_image').value
        detection_hz      = float(self.get_parameter('detection_hz').value)
        self._det_interval    = 1.0 / max(detection_hz, 0.1)
        self._last_det_time   = 0.0
        self._max_dets        = int(self.get_parameter('max_detections').value)
        self._max_poly        = int(self.get_parameter('max_poly_points').value)
        _use_half             = bool(self.get_parameter('use_half').value)
        self._use_half        = _use_half and ('cuda' in device or device.isdigit())
        self._imgsz           = int(self.get_parameter('imgsz').value)
        _torch_threads        = int(self.get_parameter('torch_threads').value)
        _use_openvino         = bool(self.get_parameter('use_openvino').value) and 'cuda' not in device and not device.isdigit()

        # Load model (auto-downloads yolov8n-seg.pt on first run)
        self.get_logger().info(f'Loading YOLO model: {model_path}  device={device}')
        from ultralytics import YOLO  # deferred import so node starts even if ultralytics missing
        if _torch_threads > 0:
            import torch
            torch.set_num_threads(_torch_threads)
            self.get_logger().info(f'torch threads capped to {_torch_threads}')

        if _use_openvino and str(model_path).endswith('.pt'):
            from pathlib import Path
            ov_dir = Path(model_path).with_suffix('').as_posix() + '_openvino_model'
            if not Path(ov_dir).exists():
                self.get_logger().info(
                    f'Exporting {model_path} -> OpenVINO IR (imgsz={self._imgsz}); first run only...'
                )
                # export is fixed-shape: imgsz here must match inference imgsz below.
                # delete the *_openvino_model dir to force re-export after changing imgsz.
                YOLO(model_path).export(format='openvino', imgsz=self._imgsz, half=False)
            self.get_logger().info(f'Loading OpenVINO model: {ov_dir}')
            self._model = YOLO(ov_dir, task='segment')
        else:
            self._model = YOLO(model_path)
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

        # Publishers
        self._det_pub = self.create_publisher(String, '/ur3/detections', 10)
        if self._pub_dbg:
            self._img_pub = self.create_publisher(Image, '/ur3/detections/image', 10)

        self.get_logger().info(f'YOLODetectorNode ready  colour={image_topic}')

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
            half=self._use_half,
            imgsz=self._imgsz,
            max_det=self._max_dets,   # cap NMS + mask gen before it runs
        )[0]

        self._last_det_time = now

        # pull all boxes off the device in one sync instead of N per-box syncs
        boxes  = results.boxes
        xyxy   = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        clss   = boxes.cls.cpu().numpy().astype(int)
        n      = min(len(xyxy), self._max_dets)
        masks_xy = results.masks.xy if (self._max_poly > 0 and results.masks is not None) else None

        detections = []
        for i in range(n):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            det = {
                'label': results.names[clss[i]],
                'conf':  round(float(confs[i]), 4),
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
            if masks_xy is not None and i < len(masks_xy):
                pts = masks_xy[i]  # (N, 2) contour pixels
                if len(pts) > self._max_poly:
                    # ceil division so the result is always <= max_poly; floor
                    # division gave step=1 (no subsample) for counts in
                    # (max_poly, 2*max_poly), overflowing the Unity overlay cap
                    step = max(1, -(-len(pts) // self._max_poly))
                    pts  = pts[::step]
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
