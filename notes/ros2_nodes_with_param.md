1. Writing a Basic Node
Comment: Initialize and run a basic ROS 2 Node.
Description: This is the foundational boilerplate for creating a ROS 2 node using Python. It involves initializing the rclpy library, creating a class that inherits from Node, and keeping it alive using spin().

Template:

Python
import rclpy
from rclpy.node import Node

class MyCustomNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        self.get_logger().info('Node has been started!')

def main(args=None):
    rclpy.init(args=args)
    node = MyCustomNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
Arguments Explanation:

'my_node_name' (string): The name of the node as it will appear in the ROS 2 network (e.g., when you type ros2 node list).

args=None (list): Command-line arguments passed to the ROS 2 context during initialization.

node (Node object): The instance of your custom node class passed to rclpy.spin(), which blocks the thread and keeps the node active to process callbacks.

2. Publisher
Comment: Send data to a specific topic.
Description: A publisher broadcasts messages of a specific data type over a defined topic name. Any node subscribed to this topic will receive the messages.

Template:

Python
from std_msgs.msg import String # Import your message type

class MyPublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(String, 'my_topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS 2!'
        self.publisher_.publish(msg)
Arguments Explanation:

String (Message Class): The type of message being published. Must match the subscriber's expected type.

'my_topic' (string): The name of the topic you are publishing to.

10 (integer/QoS Profile): The queue size (history depth). It determines how many messages to store in the buffer if the subscriber is not receiving them fast enough.

1.0 (float): The timer period in seconds (used in create_timer to trigger the callback periodically).

3. Subscriber (Listener)
Comment: Receive data from a specific topic.
Description: A subscriber listens to a specific topic and executes a callback function every time a new message is published to that topic.

Template:

Python
from std_msgs.msg import String

class MySubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'my_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
Arguments Explanation:

String (Message Class): The type of message to listen for.

'my_topic' (string): The name of the topic to subscribe to.

self.listener_callback (callable): The function that will be executed whenever a message is received.

10 (integer/QoS Profile): The queue size for incoming messages.

msg (Message Object): The actual message data passed into the callback function.

4. Parameters: Definition inside a Node
Comment: Declare and read node-specific configuration values.
Description: In ROS 2, parameters must be explicitly declared before they can be used or modified. You can assign default values during declaration.

Template:

Python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('param_node')
        
        # 1. Declare the parameter with a default value
        self.declare_parameter('my_param', 'default_string_value')
        
        # 2. Get the parameter value
        my_param_value = self.get_parameter('my_param').get_parameter_value().string_value
        
        self.get_logger().info(f'Parameter value: {my_param_value}')
Arguments Explanation:

'my_param' (string): The name of the parameter.

'default_string_value' (any standard type): The default fallback value if no parameter is provided externally.

.string_value (property): Extracts the specific Python type from the ROS 2 parameter object (other options include .integer_value, .double_value, .bool_value).

5. Parameters: Setting from Terminal
Comment: Override parameter defaults dynamically via the Command Line Interface.
Description: When running a node, you can use ROS arguments to inject parameter values without altering the Python code.

Command:

Bash
ros2 run my_package my_executable --ros-args -p my_param:="new_value"
Arguments Explanation:

my_package: The name of your ROS 2 package.

my_executable: The name of your Python executable (as defined in setup.py).

--ros-args: A flag indicating that the following arguments are meant for the ROS 2 system, not your standard Python sys.argv.

-p: Flag specifying that a parameter assignment follows.

my_param:="new_value": The key-value pair. Note the := syntax used for assignment in ROS 2.

6. Parameters: Loading from a YAML File at Initialization
Comment: Load complex configurations and multiple parameters at launch.
Description: Instead of passing many parameters via the terminal, you can define them in a YAML file and pass the file when running the node.

YAML Template (config.yaml):

YAML
/my_node_name:
  ros__parameters:1. Writing a Basic Node
Comment: Initialize and run a basic ROS 2 Node.
Description: This is the foundational boilerplate for creating a ROS 2 node using Python. It involves initializing the rclpy library, creating a class that inherits from Node, and keeping it alive using spin().

Template:

Python
import rclpy
from rclpy.node import Node

class MyCustomNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        self.get_logger().info('Node has been started!')

def main(args=None):
    rclpy.init(args=args)
    node = MyCustomNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
Arguments Explanation:

'my_node_name' (string): The name of the node as it will appear in the ROS 2 network (e.g., when you type ros2 node list).

args=None (list): Command-line arguments passed to the ROS 2 context during initialization.

node (Node object): The instance of your custom node class passed to rclpy.spin(), which blocks the thread and keeps the node active to process callbacks.

2. Publisher
Comment: Send data to a specific topic.
Description: A publisher broadcasts messages of a specific data type over a defined topic name. Any node subscribed to this topic will receive the messages.

Template:

Python
from std_msgs.msg import String # Import your message type

class MyPublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(String, 'my_topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS 2!'
        self.publisher_.publish(msg)
Arguments Explanation:

String (Message Class): The type of message being published. Must match the subscriber's expected type.

'my_topic' (string): The name of the topic you are publishing to.

10 (integer/QoS Profile): The queue size (history depth). It determines how many messages to store in the buffer if the subscriber is not receiving them fast enough.

1.0 (float): The timer period in seconds (used in create_timer to trigger the callback periodically).

3. Subscriber (Listener)
Comment: Receive data from a specific topic.
Description: A subscriber listens to a specific topic and executes a callback function every time a new message is published to that topic.

Template:

Python
from std_msgs.msg import String

class MySubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'my_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
Arguments Explanation:

String (Message Class): The type of message to listen for.

'my_topic' (string): The name of the topic to subscribe to.

self.listener_callback (callable): The function that will be executed whenever a message is received.

10 (integer/QoS Profile): The queue size for incoming messages.

msg (Message Object): The actual message data passed into the callback function.

4. Parameters: Definition inside a Node
Comment: Declare and read node-specific configuration values.
Description: In ROS 2, parameters must be explicitly declared before they can be used or modified. You can assign default values during declaration.

Template:

Python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('param_node')
        
        # 1. Declare the parameter with a default value
        self.declare_parameter('my_param', 'default_string_value')
        
        # 2. Get the parameter value
        my_param_value = self.get_parameter('my_param').get_parameter_value().string_value
        
        self.get_logger().info(f'Parameter value: {my_param_value}')
Arguments Explanation:

'my_param' (string): The name of the parameter.

'default_string_value' (any standard type): The default fallback value if no parameter is provided externally.

.string_value (property): Extracts the specific Python type from the ROS 2 parameter object (other options include .integer_value, .double_value, .bool_value).

5. Parameters: Setting from Terminal
Comment: Override parameter defaults dynamically via the Command Line Interface.
Description: When running a node, you can use ROS arguments to inject parameter values without altering the Python code.

Command:

Bash
ros2 run my_package my_executable --ros-args -p my_param:="new_value"
Arguments Explanation:

my_package: The name of your ROS 2 package.

my_executable: The name of your Python executable (as defined in setup.py).

--ros-args: A flag indicating that the following arguments are meant for the ROS 2 system, not your standard Python sys.argv.

-p: Flag specifying that a parameter assignment follows.

my_param:="new_value": The key-value pair. Note the := syntax used for assignment in ROS 2.

6. Parameters: Loading from a YAML File at Initialization
Comment: Load complex configurations and multiple parameters at launch.
Description: Instead of passing many parameters via the terminal, you can define them in a YAML file and pass the file when running the node.

YAML Template (config.yaml):

YAML
/my_node_name:
  ros__parameters:
    my_param: "value_from_yaml"
    speed_limit: 50
    is_active: true
Command:

Bash
ros2 run my_package my_executable --ros-args --params-file /path/to/config.yaml
Arguments Explanation:

/my_node_name (YAML Key): MUST match the exact node name you defined in super().__init__('my_node_name').

ros__parameters (YAML Key): A mandatory ROS 2 keyword that dictates the following dictionary contains node parameters.

--params-file: The ROS 2 argument flag telling the system to read parameters from the specified absolute or relative path to the YAML file.
    my_param: "value_from_yaml"
    speed_limit: 50
    is_active: true
Command:

Bash
ros2 run my_package my_executable --ros-args --params-file /path/to/config.yaml
Arguments Explanation:

/my_node_name (YAML Key): MUST match the exact node name you defined in super().__init__('my_node_name').

ros__parameters (YAML Key): A mandatory ROS 2 keyword that dictates the following dictionary contains node parameters.

--params-file: The ROS 2 argument flag telling the system to read parameters from the specified absolute or relative path to the YAML file.