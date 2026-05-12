import rclpy
from rclpy.node import Node

from scipy.spatial.transform import Rotation as r
import numpy as np

from nav_msgs.msg._odometry import Odometry
from cf_control_msgs.msg import ThrustAndTorque
from drone_model_msgs.msg import TrajectorySetpoint


class lee_controller(Node):
    def __init__(self):
        super().__init__('LeeController')

        # self.data_recied = False

        # --- controller configuration --- 
        self.declare_parameter('kx', 1.0)
        self.declare_parameter('kv', 1.0)
        self.declare_parameter('kR', 1.0)
        self.declare_parameter('kw', 1.0)

        self.kx = self.get_parameter('kx').value
        self.kv = self.get_parameter('kv').value
        self.kR = self.get_parameter('kR').value
        self.kw = self.get_parameter('kw').value

        # --- physical parameter ---
        self.g_vect = np.array([0.0, 0.0, -9.81])
        self.g = 9.81

        # --- drone parameters ---
        self.declare_parameter('m', 1.0)  # [kg]
        self.m = self.get_parameter('m').value

        self.declare_parameter('J', [0.0]*9) # [kg * m^2]
        J_flat = self.get_parameter('J').value
        self.J = np.array(J_flat, dtype = np.float32).reshape((3,3))

        # === init dron state ===
        self.act_timestap = 0.0
        self.act_translation = np.zeros((3,))
        self.act_linear_vel = np.zeros((3,))
        self.act_rotation_q = np.zeros((4,))
        self.act_rotation_q[-1] = 1.0
        self.act_angular_vel = np.zeros((3,))

        self.x_sp = np.zeros((3,))
        self.v_sp = np.zeros((3,))
        self.a_sp = np.zeros((3,))
        self.y_sp = 0.0
        self.w_sp = np.zeros((3,))
        self.dw_sp = np.zeros((3,))

        # === ROS publishers and subscribers ===
        self.declare_parameter('dt', 0.01)
        self.dt = self.get_parameter('dt').value
        self.physics_timer = self.create_timer(self.dt, self.set_output)

        # drone state subsriber 
        self.declare_parameter('ROS2_topic_name_state', '/crazyflie/Odometry') 
        ROS2_topic_name_state = self.get_parameter('ROS2_topic_name_state').value

        self.subscriber_state = self.create_subscription(
            Odometry, ROS2_topic_name_state, self.get_input, 10
        )

        # drone setpoint subscriber 
        self.declare_parameter('ROS2_topic_name_setpoint', '/crazyflie/Setpoints') 
        ROS2_topic_name_setpoint = self.get_parameter('ROS2_topic_name_setpoint').value

        self.subscriber_setpoint = self.create_subscription(
            TrajectorySetpoint, ROS2_topic_name_setpoint, self.setpoint, 10
        )

        # drone ThrustAndTorque publisher 
        self.declare_parameter('ROS2_topic_name_ThrustAndTorque', '/cf_control/control_command') 
        ROS2_topic_name_thrustandtorque = self.get_parameter('ROS2_topic_name_ThrustAndTorque').value

        self.publisher = self.create_publisher(ThrustAndTorque, ROS2_topic_name_thrustandtorque, 10)


    def setpoint(self, msg):
        """
        Setting actual setpoints for Lee Controller 
        """
        if msg is not None:
            self.x_sp = msg.x  # displacement, [3,]
            self.v_sp = msg.v  # linear velocity, [3,], set to 0 (?)
            self.a_sp = msg.a  # linear acceleration [3,], set to 0 (?)
            self.y_sp = msg.y  # yaw, [1,]
            self.w_sp = msg.w  # yaw velocity, set to 0 (?)
            self.dw_sp = msg.dw # msg.dw  # angular acceleration, [3, ], set to 0 (?)

    def output(self, x, v, q, w):
        """
        Calculates output (thrust, torque) and publsih it 
        """
        ax_z_g = np.array([0, 0, 1])  # z axis in global ref frame

        # === error: position and linear velocity ===
        self.get_logger().info(f'[debug] x: act = {x}, sp = {self.x_sp}')
        ex = x - self.x_sp  # position error [3,]
        ev = v - self.v_sp  # velocity error [3,]

        # === Target thrust - global body frame === 
        

        T_target = (
            -self.kx * ex - self.kv * ev + ax_z_g * self.m * self.g + self.m * self.a_sp
        )  # desire thrust vector
        
        if T_target[2] < 0.01:
            T_target[2] = 0.01
        # === compose target rotation matrix ===

        # z axis of body frame in global reference frame
        
        if np.linalg.norm(T_target) > 1e-4:
            b3 = T_target / np.linalg.norm(T_target)  # z axis of body frame
        else:
            b3 = ax_z_g  

        self.get_logger().info(f'[debug] ex = {ex}, ev = {ev}, T_target = {T_target}, b3: {b3}')
        # x axis of body frame in global reference frame - set by yaw setpoint
        b1 = np.array([np.cos(self.y_sp), np.sin(self.y_sp), 0])  # x axis of body frame

        # y axis of body frame in global reference frame - forced by x and z axis
        b2 = np.cross(b3, b1)  # y axis of body frame
        b2 = b2 / np.linalg.norm(b2)

        # correction for b1 axis - make sure, that b1 is perpendicular to b2 and b3
        b1 = np.cross(b2, b3)  # ???

        # compose axis into rotation matrix
        R_target_matrix = np.column_stack((b1, b2, b3))
        R_target = r.from_matrix(R_target_matrix)

        # === rotation error ===
        R_actual = r.from_quat(q)

        R_target_mat = R_target.as_matrix()
        R_actual_mat = R_actual.as_matrix()

        RdTR = R_target_mat.T @ R_actual_mat
        RTRd = R_actual_mat.T @ R_target_mat
        eR = 0.5 * vee_map(RdTR - RTRd)

        # angular velocity error
        # ew = w - (R_actual.inv() * R_target * self.w_sp)
        ew = w - (R_actual.inv() * R_target).apply(self.w_sp)

        # === compute control output ===

        # required forces in global frame transformed to local frame,
        # and casted on vertical axis, shape [3, ]
       
        F = np.dot(T_target, R_actual.apply(ax_z_g))

        R_err_mat = (R_actual.inv() * R_target).as_matrix()
        
        M = (
            -self.kR * eR
            - self.kw * ew
            + np.cross(w, (self.J @ w))
            - self.J
            @ (
                hat(w) @ R_err_mat @ self.w_sp
                - R_err_mat @ self.dw_sp  
            )
        )
        return F, M

    def get_input(self, msg: Odometry):
        '''
        Receive actual drone state 
        '''
        if msg is not None:
          
            pos = msg.pose.pose.position
            self.act_translation = np.array([pos.x, pos.y, pos.z])
            
            vel = msg.twist.twist.linear
            self.act_linear_vel = np.array([vel.x, vel.y, vel.z])
            
            quat = msg.pose.pose.orientation
            self.act_rotation_q = np.array([quat.x, quat.y, quat.z, quat.w])
            
            ang_vel = msg.twist.twist.angular
            self.act_angular_vel = np.array([ang_vel.x, ang_vel.y, ang_vel.z])




    def set_output(self):
            msg = ThrustAndTorque()

            thrust, torque = self.output(
                self.act_translation,
                self.act_linear_vel,
                self.act_rotation_q,
                self.act_angular_vel,
            )
            # print(thrust.shape, torque.shape)

            msg.collective_thrust = float(thrust)
            msg.torque.x = float(torque[0])
            msg.torque.y = float(torque[1])
            msg.torque.z = float(torque[2])
            self.publisher.publish(msg)

            # self.get_logger().info(f'Thrust: {float(thrust):.4}, Torque: [{torque[0]:.4}, {torque[1]:.4}, {torque[2]:.4}]')
        

def vee_map(x):
    """
    skew-symmetric -> column vector
    """
    return np.array([x[2, 1], x[0, 2], x[1, 0]])


def hat(x):
    """
    columnt vector -> skew-symmetric
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def main(args=None):
    rclpy.init(args=args)

    node = lee_controller()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
