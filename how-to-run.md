## RUN PACKAGE
___
simulation

```bash
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py
```
___
gazebo bridge 

```bash
ros2 run ros_gz_bridge parameter_bridge '/model/crazyflie/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' --ros-args -r /model/crazyflie/odometry:=/crazyflie/ActualState
```
___
thrust allocation

```bash
ros2 run cf_control mixer
```
___
trajectory generator 

```bash
ros2 run trajectory trajectory_generator --ros-args --params-file /home/developer/ros2_ws/src/trajectory/trajectory/param.yaml
```
___
lee controller 

```bash
ros2 run controller lee_controller --ros-args --params-file /home/developer/ros2_ws/src/controller/controller/param.yaml
```
___
mpc controller 

```bash
ros2 run mpc_controller mpc_controller --ros-args --params-file /home/developer/ros2_ws/src/mpc_controller/mpc_controller/param.yaml
```

Ruvchomic XLaunch (z parametrem 0)

wewnątrz konenetra wywołać: 
export DISPLAY=host.docker.internal:0.0

ros2 bag record /crazyflie/ActualState

