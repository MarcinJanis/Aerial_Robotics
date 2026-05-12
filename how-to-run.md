

## RUN PACKAGE

```bash
ros2 run controller lee_controller --ros-args --params-file /home/developer/ros2_ws/src/controller/controller/param.yaml
```

```bash
ros2 run trajectory trajectory_generator --ros-args --params-file /home/developer/ros2_ws/src/trajectory/trajectory/param.yaml
```

```bash
ros2 launch ros_gz_crazyflie_bringup crazyflie_simulation.launch.py
```

```bash
ros2 run cf_control mixer
```
