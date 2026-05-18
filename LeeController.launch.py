import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # --- simulation --- 
    # force_headless = SetEnvironmentVariable('QT_QPA_PLATFORM', 'offscreen')

   
    # crazyflie_bringup_dir = get_package_share_directory('ros_gz_crazyflie_bringup')
    
    # crazyflie_sim_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(crazyflie_bringup_dir, 'launch', 'crazyflie_simulation.launch.py')
    #     )
    # )

    # Node - Controller 
    lee_controller_node = Node(
        package='controller',
        executable='lee_controller',
        output='screen', 

        parameters=[
            '/home/developer/ros2_ws/src/controller/controller/param.yaml'
        ]
    )

    # Node - Trajectory
    trajectory_node = Node(
        package='trajectory',
        executable='trajectory_generator',
        output='screen',
        parameters=[
            '/home/developer/ros2_ws/src/trajectory/trajectory/param.yaml'
        ]
    )

    # Node - Thrust allocation
    mixer_node = Node(
        package='cf_control',
        executable='mixer',
        output='screen'
    )

    # Bridge between gazebo and ros2
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=[
            # <Nazwa_Topiku_W_Gazebo>@<Typ_ROS>[<Typ_Gazebo>
            # [ - direction -> 
            '/model/crazyflie/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        ],
        remappings=[
            ('/model/crazyflie/odometry', '/crazyflie/ActualState')
        ],
        output='screen'
    )

    return LaunchDescription([
        # crazyflie_sim_launch,
        bridge_node,
        lee_controller_node,
        trajectory_node,
        mixer_node
    ])
