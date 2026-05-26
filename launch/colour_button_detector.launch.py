import json
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Default button colour definitions.
# Red wraps around hue=0 in OpenCV HSV (0-179), so two entries share the
# same label - they are OR-ed together before contour finding.
# Tune h_lo/h_hi/s_lo/v_lo per your lighting and button colours.
_DEFAULT_BUTTONS = json.dumps([
    {"label": "red",   "h_lo":   0, "h_hi":  10, "s_lo": 80, "v_lo": 60},
    {"label": "red",   "h_lo": 170, "h_hi": 180, "s_lo": 80, "v_lo": 60},
    {"label": "green", "h_lo":  40, "h_hi":  80, "s_lo": 80, "v_lo": 60},
    {"label": "blue",  "h_lo": 100, "h_hi": 130, "s_lo": 80, "v_lo": 60},
])


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/camera/color/image_raw',
            description='Colour image topic from the D455.',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw',
            description='Aligned depth image topic from the D455.',
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/camera/color/camera_info',
            description='Camera info topic for intrinsics.',
        ),
        DeclareLaunchArgument(
            'buttons',
            default_value=_DEFAULT_BUTTONS,
            description=(
                'JSON array of button colour definitions. '
                'Each entry: {"label":"<name>","h_lo":<0-179>,"h_hi":<0-179>,'
                '"s_lo":<0-255>,"v_lo":<0-255>}. '
                'Define two entries with the same label to handle hue-wrapped '
                'colours (e.g. red at 0-10 and 170-180).'
            ),
        ),
        DeclareLaunchArgument(
            'min_area',
            default_value='200',
            description='Minimum blob area in pixels. Smaller blobs are ignored.',
        ),
        DeclareLaunchArgument(
            'max_area',
            default_value='50000',
            description='Maximum blob area in pixels. Larger blobs are ignored.',
        ),
        DeclareLaunchArgument(
            'detection_hz',
            default_value='15.0',
            description='Maximum detection publish rate (Hz).',
        ),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='false',
            description='Publish annotated image to /ur3/button_detections/image.',
        ),
        DeclareLaunchArgument(
            'use_cuda',
            default_value='true',
            description=(
                'Use OpenCV CUDA for BGR→HSV conversion when available. '
                'Falls back to CPU automatically if CUDA is not detected.'
            ),
        ),

        Node(
            package='ur3_perception',
            executable='colour_button_detector',
            name='colour_button_detector',
            output='screen',
            parameters=[{
                'image_topic':         LaunchConfiguration('image_topic'),
                'depth_topic':         LaunchConfiguration('depth_topic'),
                'camera_info_topic':   LaunchConfiguration('camera_info_topic'),
                'buttons':             LaunchConfiguration('buttons'),
                'min_area':            LaunchConfiguration('min_area'),
                'max_area':            LaunchConfiguration('max_area'),
                'detection_hz':        LaunchConfiguration('detection_hz'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'use_cuda':            LaunchConfiguration('use_cuda'),
            }],
        ),
    ])
