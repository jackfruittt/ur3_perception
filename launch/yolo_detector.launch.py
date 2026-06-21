from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='yolov8n-seg.pt',
            description='Path to YOLO weights file (.pt or .onnx). '
                        'yolov8n-seg.pt is auto-downloaded on first run.',
        ),
        DeclareLaunchArgument(
            'confidence',
            default_value='0.40',
            description='Minimum detection confidence threshold (0.0 – 1.0).',
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cpu',
            description='Inference device: "cpu", "cuda:0", or GPU index "0".',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/camera/color/image_raw',
            description='ROS image topic from the D455.',
        ),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='false',
            description='Publish annotated image to /ur3/detections/image.',
        ),
        DeclareLaunchArgument(
            'imgsz',
            default_value='480',
            description='Inference resolution (longest side). Lower = faster on CPU. '
                        'Independent of the 640x480 stream sent to Unity.',
        ),
        DeclareLaunchArgument(
            'torch_threads',
            default_value='0',
            description='Cap torch CPU threads (0 = leave default). Set to share cores with Unity/RViz.',
        ),
        DeclareLaunchArgument(
            'use_openvino',
            default_value='true',
            description='CPU only: auto-export and load OpenVINO IR (~2-3x on Intel). Ignored on CUDA.',
        ),

        Node(
            package='ur3_perception',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'image_topic':         LaunchConfiguration('image_topic'),
                'model_path':          LaunchConfiguration('model_path'),
                'confidence':          LaunchConfiguration('confidence'),
                'device':              LaunchConfiguration('device'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'imgsz':               LaunchConfiguration('imgsz'),
                'torch_threads':       LaunchConfiguration('torch_threads'),
                'use_openvino':        LaunchConfiguration('use_openvino'),
            }],
        ),
    ])
