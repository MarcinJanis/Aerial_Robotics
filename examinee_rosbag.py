import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import matplotlib.pyplot as plt 
# Path to the ROS2 bag directory (contains the .db3 and metadata.yaml)


def convert_rosbag_to_numpy(bag_path_str: str):
    bag_dir = Path(bag_path_str)
    typestore = get_typestore(Stores.LATEST)

    topics_data = {}

    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        for connection in reader.connections:
            if connection.topic not in topics_data:
                topics_data[connection.topic] = []

        for connection, timestamp, rawdata in reader.messages():
            topic_name = connection.topic
            msg_type = connection.msgtype

            try:
                msg = reader.deserialize(rawdata, msg_type)
            except Exception as e:
                continue

            # --- TARGETED ODOMETRY EXTRACTION ---
            if msg_type == "nav_msgs/msg/Odometry":
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                # Save as a flat list: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w]
                val = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

            # --- Fallbacks for other possible topics ---
            elif hasattr(msg, "data"):
                val = msg.data
            elif hasattr(msg, "position"):
                val = msg.position
            else:
                val = str(msg)

            topics_data[topic_name].append(val)

    topic_names_list = []
    numpy_arrays_list = []

    for topic, data in topics_data.items():
        if data:
            topic_names_list.append(topic)
            numpy_arrays_list.append(np.array(data))

    return topic_names_list, numpy_arrays_list

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    bag_dir = Path('C:/Users/janis/OneDrive/Pulpit/Studia/Semestr_III/Aerial_Robotics/rosbag2_2026_05_18-21_12_23')

    # Call the function
    topics, arrays = convert_rosbag_to_numpy(bag_dir)

    fig = plt.figure(figsize=(10, 8))
    start = 0 
    end = 300
    for name, arr in zip(topics, arrays):
        print(name, ': ', arr.shape) 
     
        ax = fig.add_subplot(111, projection="3d")  

        ax.plot(arr[start:end, 0], arr[start:end, 1], arr[start:end, 2], label="Drone Path", color="blue", lw=2, alpha=0.8)

    plt.show()