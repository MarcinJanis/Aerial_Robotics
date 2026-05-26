import rclpy
from rclpy.node import Node

from scipy.spatial.transform import Rotation as r
import numpy as np
import math

from nav_msgs.msg._odometry import Odometry
from cf_control_msgs.msg import ThrustAndTorque

from drone_model_msgs.msg import PolynomialTrajectory


class lee_controller(Node):
    def __init__(self):
        super().__init__('LeeController')

        # --- controller configuration ---
        # 
        self.declare_parameter('kx', 1.0)
        self.declare_parameter('kv', 1.0)
        self.declare_parameter('kR', [0.004, 0.004, 0.0002])
        self.declare_parameter('kw', [0.0004, 0.0004, 0.00002])

        self.kx = self.get_parameter('kx').value
        self.kv = self.get_parameter('kv').value
        self.kR = np.array(self.get_parameter('kR').value)
        self.kw = np.array(self.get_parameter('kw').value)

        # --- physical parameter ---
        self.g_vect = np.array([0.0, 0.0, -9.81])
        self.g = 9.81
        
        self.declare_parameter('max_thrust', 0.4) 
        self.max_thrust_CF = self.get_parameter('max_thrust').value
        
        # --- drone parameters ---
        self.declare_parameter('m', 1.0)  # [kg]
        self.m = self.get_parameter('m').value

        self.declare_parameter('J', [0.0]*9) # [kg * m^2]
        J_flat = self.get_parameter('J').value
        self.J = np.array(J_flat, dtype = np.float32).reshape((3,3))

        # === init dron state ===
        self.act_translation = np.zeros((3,))
        self.act_linear_vel = np.zeros((3,))
        self.act_rotation_q = np.zeros((4,))
        self.act_rotation_q[-1] = 1.0
        self.act_angular_vel = np.zeros((3,))

        self.prev_time_stamp = 0.0 # prev msg time stamp to derivative of position
        self.vel_lp_filter = 0.2
        self.prev_time_stamp = 0.0
        self.prev_pose = act_translation
        
        # === trajectory state (Polynomials) ===
        self.poly_x = np.zeros(8)
        self.poly_y = np.zeros(8)
        self.poly_z = np.zeros(8)
        self.poly_T = 1.0               
        self.poly_t_local_start = 0.0   # Local time wien msg received
        self.poly_start_time_ros = None # ROS timer staten when msg recevied
        
        self.last_known_yaw = 0.0

        # === controller setpoints ===
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

        # drone setpoint subscriber - aktualizacja na PolynomialTrajectory
        self.declare_parameter('ROS2_topic_name_setpoint', '/crazyflie/Trajectory') 
        ROS2_topic_name_setpoint = self.get_parameter('ROS2_topic_name_setpoint').value

        self.subscriber_setpoint = self.create_subscription(
            PolynomialTrajectory, ROS2_topic_name_setpoint, self.setpoint, 10
        )

        # drone ThrustAndTorque publisher 
        self.declare_parameter('ROS2_topic_name_ThrustAndTorque', '/cf_control/control_command') 
        ROS2_topic_name_thrustandtorque = self.get_parameter('ROS2_topic_name_ThrustAndTorque').value

        self.publisher = self.create_publisher(ThrustAndTorque, ROS2_topic_name_thrustandtorque, 10)


    def setpoint(self, msg: PolynomialTrajectory):
        """
        Pobiera najnowsze wielomiany i zapisuje dokładny czas ich nadejścia,
        aby można było je płynnie interpolować w set_output()
        """
        if msg is not None:
            self.poly_x = np.array(msg.polynomial_x)
            self.poly_y = np.array(msg.polynomial_y)
            self.poly_z = np.array(msg.polynomial_z)
            self.poly_T = msg.duration # duration of segment 
            
            # local time (from start of this segment)
            self.poly_t_local_start = msg.timestamp
            
            # get ROS timer state when msg recived 
            self.poly_start_time_ros = self.get_clock().now()

    def sample_polynomial(self, c, s, T):
        """ Próbkuje wielomian dla znormalizowanego czasu s w [0, 1] """
        if T < 1e-4:
            return c[0], 0.0, 0.0
        
        
        s = np.clip(s, 0.0, 1.0) # clamp as s is normalized vector (0 - 1)
        
        # Pose - 0 derivative
        p = c[0] + c[1]*s + c[2]*s**2 + c[3]*s**3 + c[4]*s**4 + c[5]*s**5 + c[6]*s**6 + c[7]*s**7
        
        # Velocity - 1st derivative
        v_s = c[1] + 2*c[2]*s + 3*c[3]*s**2 + 4*c[4]*s**3 + 5*c[5]*s**4 + 6*c[6]*s**5 + 7*c[7]*s**6
        v = v_s / T
        
        # Acceleration - 3rd derivative
        a_s = 2*c[2] + 6*c[3]*s + 12*c[4]*s**2 + 20*c[5]*s**3 + 30*c[6]*s**4 + 42*c[7]*s**5
        a = a_s / (T**2)

        return p, v, a

    def output(self, x, v, q, w):
        """
        Calculates output (thrust, torque) and publsih it 
        """
        ax_z_g = np.array([0, 0, 1])  # z axis in global ref frame

        # === error: position and linear velocity ===
        ex = x - self.x_sp  # position error [3,]
        ev = v - self.v_sp  # velocity error [3,]

        # === Target thrust - global body frame === 
        T_target = (
            -self.kx * ex - self.kv * ev + ax_z_g * self.m * self.g + self.m * self.a_sp
        )  # desire thrust vector
        
        if T_target[2] < 0.01:
            T_target[2] = 0.01
            
        # === compose target rotation matrix ===
        if np.linalg.norm(T_target) > 1e-4:
            b3 = T_target / np.linalg.norm(T_target)  # z axis of body frame
        else:
            b3 = ax_z_g  

        # x axis of body frame in global reference frame - set by yaw setpoint
        b1 = np.array([np.cos(self.y_sp), np.sin(self.y_sp), 0])  # x axis of body frame

        # y axis of body frame in global reference frame - forced by x and z axis
        b2 = np.cross(b3, b1)  # y axis of body frame
        b2 = b2 / np.linalg.norm(b2)

        # correction for b1 axis - make sure, that b1 is perpendicular to b2 and b3
        b1 = np.cross(b2, b3)  

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
        ew = w - (R_actual.inv() * R_target).apply(self.w_sp)

        # === compute control output ===
        F = np.dot(T_target, R_actual.apply(ax_z_g))
        F = np.clip(F, 0.01, self.max_thrust_CF) 

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

    def get_input_2(self, msg: Odometry):
        '''
        Receive actual drone state 
        '''
        if msg is not None:
            # pose - global 
            pos = msg.pose.pose.position
            self.act_translation = np.array([pos.x, pos.y, pos.z])
            
            # orientation 
            quat = msg.pose.pose.orientation
            self.act_rotation_q = np.array([quat.x, quat.y, quat.z, quat.w])
            R_actual = r.from_quat(self.act_rotation_q)
            
            # velocity -> transform from body to global frame
            #--- velocity - ver 1: from Gazebo
            vel_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            self.act_linear_vel = R_actual.apply(vel_body) 
            #--- velocity - ver 2: based on pose
            act_time_stamp = msg.pose.timestamp # check !!!
            d_pose = (self.prev_pose - self.act_translation) - np.max(act_time_stamp - self.prev_time_stamp, 1e-4)
            d_pose_filtered = d_pose - self.vel_lp_filter * self.act_linear_vel # self.act_linear_vel used as prev 
            
            self.act_linear_vel = d_pose_filtered
            self.prev_time_stamp = act_time_stamp
            self.prev_pose = self.act_translation
            
            #---
            
            # angular velocity 
            ang_vel = msg.twist.twist.angular
            self.act_angular_vel = np.array([ang_vel.x, ang_vel.y, ang_vel.z])

    def set_output(self):
        # --- Inerpolate polynomials ---
        if self.poly_start_time_ros is not None:
            now = self.get_clock().now()
            dt_since_msg = (now - self.poly_start_time_ros).nanoseconds / 1e9
            
            t_local = self.poly_t_local_start + dt_since_msg
            T = max(self.poly_T, 1e-4) 
            s = t_local / T

            # Sample
            p_x, v_x, a_x = self.sample_polynomial(self.poly_x, s, T)
            p_y, v_y, a_y = self.sample_polynomial(self.poly_y, s, T)
            p_z, v_z, a_z = self.sample_polynomial(self.poly_z, s, T)

            self.x_sp = np.array([p_x, p_y, p_z])
            self.v_sp = np.array([v_x, v_y, v_z])

            # self.a_sp = np.array([a_x, a_y, a_z])
            self.a_sp = np.array([0.0, 0.0, 0.0])

            # Calc yaw and yaw rate based on x, y of trajectory
            speed_sq = v_x**2 + v_y**2
            if speed_sq > 0.5:
                yaw = math.atan2(v_y, v_x)
                yaw_rate = (v_x * a_y - v_y * a_x) / speed_sq
                self.last_known_yaw = yaw
            else:
                yaw = self.last_known_yaw
                yaw_rate = 0.0

            self.y_sp = yaw
            self.w_sp = np.array([0.0, 0.0, yaw_rate])
            self.dw_sp = np.zeros(3)

            self.y_sp = 0.0
            self.w_sp = np.array([0.0, 0.0, 0.0])
            self.dw_sp = np.zeros(3)

        # self.get_logger().info(f'State:\nActual: pose: {self.act_translation} Setpoint: {self.x_sp}')
        
        # --- Controller calculation --- 
        

        thrust, torque = self.output(
            self.act_translation,
            self.act_linear_vel,
            self.act_rotation_q,
            self.act_angular_vel,
        )
        
        # --- Publish ---
        msg = ThrustAndTorque()
        msg.collective_thrust = float(thrust)
        msg.torque.x = float(torque[0])
        msg.torque.y = float(torque[1])
        msg.torque.z = float(torque[2])
        self.publisher.publish(msg)

        # self.get_logger().info(f'Control: thrust: {thrust:.4} torque: {torque}')
        self.get_logger().info(
            f'\n--- TELEMETRIA OSI ---'
            f'\n[X] Pozycja: Akt={self.act_translation[0]:.3f}, Zad={self.x_sp[0]:.3f} | Prędkość: Akt={self.act_linear_vel[0]:.3f}, Zad={self.v_sp[0]:.3f} | Przysp Zad={self.a_sp[0]:.3f}'
            f'\n[Y] Pozycja: Akt={self.act_translation[1]:.3f}, Zad={self.x_sp[1]:.3f} | Prędkość: Akt={self.act_linear_vel[1]:.3f}, Zad={self.v_sp[1]:.3f} | Przysp Zad={self.a_sp[1]:.3f}'
            f'\n[Z] Pozycja: Akt={self.act_translation[2]:.3f}, Zad={self.x_sp[2]:.3f} | Prędkość: Akt={self.act_linear_vel[2]:.3f}, Zad={self.v_sp[2]:.3f} | Przysp Zad={self.a_sp[2]:.3f}'
            f'\n[Thrust]: {thrust:.4f} N'
        )

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
