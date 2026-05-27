# Author: Jackson Russell
#
# hsv_tuner.py
# Interactive HSV threshold tuner for colour_button_detector.py.
# Subscribes to a colour image topic and shows a cv2 window with trackbars
# for H_lo, H_hi, S_lo, S_hi, V_lo, V_hi.  The left half of the window
# shows the live colour frame; the right half shows the binary mask.
#
# Tune one colour at a time:
#
#   python3 hsv_tuner.py --label red   --topic /camera/camera/color/image_raw
#   python3 hsv_tuner.py --label green --topic /camera/camera/color/image_raw
#
# Works against live topics or a bag replay:
#
#   ros2 bag play button_tune --loop &
#   python3 hsv_tuner.py --label red
#
# Keyboard shortcuts:
#   s  - print current values as JSON ready to paste into the detector
#   c  - print H/S/V stats (mean, min, max) of the masked region (helps with
#        initial placement of sliders)
#   q  - quit
#
# Output JSON format (matches colour_button_detector.py buttons parameter):
#   {"label": "red", "h_lo": 0, "h_hi": 10, "s_lo": 80, "v_lo": 60, "v_hi": 240}
#
# Red wraps around hue=0.  Run twice (h_lo=0..10, h_lo=170..180) and combine
# both entries in your buttons list.

import argparse
import sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

_WINDOW = 'HSV Tuner'
_INITIAL = dict(h_lo=0, h_hi=179, s_lo=50, s_hi=255, v_lo=50, v_hi=255)


def _nothing(_):
    pass


class HsvTunerNode(Node):
    def __init__(self, label: str, topic: str, clahe: bool):
        super().__init__('hsv_tuner')
        self._label  = label
        self._bridge = CvBridge()
        self._frame  = None
        self._clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(f'Subscribed to {topic}  label={label}')

        # cv2 window + trackbars
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW, 1280, 480)
        for name, val in _INITIAL.items():
            hi = 179 if name in ('h_lo', 'h_hi') else 255
            cv2.createTrackbar(name, _WINDOW, val, hi, _nothing)

        # h_hi starts at max; h_lo at 0
        cv2.setTrackbarPos('h_hi', _WINDOW, 179)
        cv2.setTrackbarPos('v_hi', _WINDOW, 255)
        cv2.setTrackbarPos('s_hi', _WINDOW, 255)

        self.create_timer(0.033, self._tick)  # ~30 Hz redraw

    def _cb(self, msg: Image):
        try:
            self._frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)

    def _get_sliders(self):
        return {k: cv2.getTrackbarPos(k, _WINDOW) for k in _INITIAL}

    def _tick(self):
        if self._frame is None:
            return

        bgr = self._frame.copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        if self._clahe is not None:
            h_ch, s_ch, v_ch = cv2.split(hsv)
            v_ch = self._clahe.apply(v_ch)
            hsv  = cv2.merge([h_ch, s_ch, v_ch])

        s = self._get_sliders()
        lo   = np.array([s['h_lo'], s['s_lo'], s['v_lo']], dtype=np.uint8)
        hi   = np.array([s['h_hi'], s['s_hi'], s['v_hi']], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        # overlay mask contours on colour frame
        overlay = bgr.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)

        # HUD: slider values
        txt = (f"label={self._label}  "
               f"H:[{s['h_lo']},{s['h_hi']}]  "
               f"S:[{s['s_lo']},{s['s_hi']}]  "
               f"V:[{s['v_lo']},{s['v_hi']}]  "
               f"  s=save  c=stats  q=quit")
        cv2.putText(overlay, txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # right panel: mask as BGR
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([overlay, mask_bgr])
        cv2.imshow(_WINDOW, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting.')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord('s'):
            self._print_json()
        elif key == ord('c'):
            self._print_stats(hsv, mask)

    def _print_json(self):
        s = self._get_sliders()
        entry = {
            'label': self._label,
            'h_lo':  s['h_lo'], 'h_hi': s['h_hi'],
            's_lo':  s['s_lo'],
            'v_lo':  s['v_lo'], 'v_hi': s['v_hi'],
        }
        print('\n--- JSON entry (paste into buttons parameter) ---')
        print(json_line(entry))
        print('-------------------------------------------------\n')

    def _print_stats(self, hsv: np.ndarray, mask: np.ndarray):
        if mask.sum() == 0:
            print('[stats] mask is empty - nothing detected')
            return
        pixels = hsv[mask > 0]   # shape (N, 3)
        h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        print(f'\n--- HSV stats for masked region ({len(h)} pixels) ---')
        print(f'  H  min={h.min():3d}  max={h.max():3d}  mean={h.mean():.1f}')
        print(f'  S  min={s.min():3d}  max={s.max():3d}  mean={s.mean():.1f}')
        print(f'  V  min={v.min():3d}  max={v.max():3d}  mean={v.mean():.1f}')
        print('-----------------------------------------------------\n')


def json_line(d: dict) -> str:
    import json
    return json.dumps(d)


def main():
    parser = argparse.ArgumentParser(description='Interactive HSV tuner for colour_button_detector')
    parser.add_argument('--label', default='button',
                        help='Colour label, e.g. red / green / blue')
    parser.add_argument('--topic', default='/camera/camera/color/image_raw',
                        help='ROS image topic to subscribe to')
    parser.add_argument('--clahe', action='store_true', default=True,
                        help='Apply CLAHE to V channel before thresholding (matches detector default)')
    parser.add_argument('--no-clahe', dest='clahe', action='store_false')

    # rclpy passes extra ROS args after --ros-args; strip them for argparse
    argv = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    args = parser.parse_args(argv)

    rclpy.init()
    node = HsvTunerNode(args.label, args.topic, args.clahe)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
