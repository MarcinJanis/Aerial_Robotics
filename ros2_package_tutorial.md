# How to create new ROS2 package 


## 1. Auto init

- For package with custom msgs type (C++) execute: 

`ros2 pkg create --build-type ament_cmake <new_package_name>`

It create folder with template


- For paclage with node source code (python) execute: 

`ros2 pkg create --build-type ament_python <new_package_name> --dependencies rclpy cf_control_msgs drone_model_msgs geometry_msgss`

(after --dependecies we can put all necessery dependencies)

## 2. Fill all necessery things in c++ folder
 
to create custom msg create folder /msg, and file messange_name.msg

Add to CMakeList.txt:


`set(msg_files
  "msg/DroneState.msg"
)`

`rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)`

## 3. Fill all necessey thing in python package: 

add files .py and in setup.py add entry points:

# ... reszta pliku setup.py ...
    entry_points={
        'console_scripts': [
            # 'nazwa_ktora_wpisujesz_w_terminalu = nazwa_pakietu.nazwa_pliku:funkcja_startowa'
            'drone_sim = aerial_core.drone_node:main',
        ],
    },
)
