import rclpy
from rclpy.node import Node

import scipy 
from scipy.spatial.transform import Rotation as r
import numpy as np
import math 
import pandas as pd 
from scipy.optimize import minimize

from drone_model_msgs.msg import TrajectorySetpoint


class MinimumSnapTrajectory(Node):
    def __init__(self):

        super().__init__('MinSnapTraj')

        self.start_time = None

        # --- trajectory waypoints --- 
        self.declare_parameter('waypoints_pth', '/')
        self.waypoints_pth = self.get_parameter('waypoints_pth').value
       
        # df = pd.read_csv(self.waypoints_pth)
        # self.waypoints = df.to_numpy() 
        self.waypoints = np.loadtxt(self.waypoints_pth, delimiter=',', skiprows=1, dtype=float)
        
        self.declare_parameter('desire_speed', 1.0)
        self.speed = self.get_parameter('desire_speed').value

        # === ROS publishers and subscribers ===
        self.declare_parameter('dt', 0.01)
        self.dt = self.get_parameter('dt').value

        
        # drone ThrustAndTorque publisher 
        self.declare_parameter('ROS2_topic_name_TrajectorySetpoint', '/crazyflie/Setpoints') 
        ROS2_topic_name_TrajectorySetpoint = self.get_parameter('ROS2_topic_name_TrajectorySetpoint').value

        self.publisher = self.create_publisher(TrajectorySetpoint, ROS2_topic_name_TrajectorySetpoint, 10)

        # calculate trajectory 
        self.trajectory_coefficients, self.global_time_vect, self.local_time_vect  = self.find_trajectory()
        self.global_time_max = self.global_time_vect[-1]

    
        self.last_known_yaw = 0.0

        self.physics_timer = self.create_timer(self.dt, self.set_output)



    def set_output(self):
       
        # Get actual global time
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        

        now = self.get_clock().now()
        global_t = (now - self.start_time).nanoseconds / 1e9

        if global_t >= self.global_time_max:
            segment_idx = len(self.local_time_vect) - 1
            local_t = self.local_time_vect[-1] 
        else:
            segment_idx = np.searchsorted(self.global_time_vect, global_t)

        if segment_idx == 0:
            local_t = global_t
        else:
            local_t = global_t - self.global_time_vect[segment_idx - 1]
            
        # --- sample trajectory ---
        
        p_x, v_x, a_x = self.sample_trajectory(self.trajectory_coefficients[segment_idx, 0], local_t, global_t)
        p_y, v_y, a_y = self.sample_trajectory(self.trajectory_coefficients[segment_idx, 1], local_t, global_t)
        p_z, v_z, a_z = self.sample_trajectory(self.trajectory_coefficients[segment_idx, 2], local_t, global_t)


        # calculate yaw angle
        speed_sq = v_x**2 + v_y**2
        
        if speed_sq > 1e-4:
            yaw = math.atan2(v_y, v_x)
            yaw_rate = (v_x * a_y - v_y * a_x) / speed_sq
        else:
            yaw = self.last_known_yaw 
            yaw_rate = 0.0

        self.last_known_yaw = yaw

        self.get_logger().info(f'local time: {local_t}, global_time{global_t}, setpoint: {p_x}, {p_y}, {p_z}')
        msg = TrajectorySetpoint()

        msg.x = [float(p_x), float(p_y), float(p_z)]
        msg.v = [float(v_x), float(v_y), float(v_z)]
        msg.a = [float(a_x), float(a_y), float(a_z)]
        
        msg.y = yaw
        msg.w = [0.0, 0.0, yaw_rate]
        msg.dw = [0.0, 0.0, 0.0]

        self.publisher.publish(msg)
 


    def allocate_time(self, speed=1.0):
        '''
        allocate time vector based on distance between each poitns
        '''
        dist = np.linalg.norm(self.waypoints[1:, :] - self.waypoints[:-1, :], axis=-1)
        time_vect = dist / speed # v = x/t -> t = x/v
        return time_vect

    
    def compute_Q_matrix(self, T, degree=7):
        
        '''
        Calculates the 8x8 Hessian matrix Q for the minimum snap cost function.
        T: time duration of this specific segment.

        Q matrix, used in equation:
        $$c^TQc$$ is equal to Integral of squared, 4th derivative from 7th degree polynomial.
        so loss function can be describe either as:
        $$J = \int{{p^{IV}(t)}^2}dt$$
        or:
        $$J = c^TQc$$
        (somehow!)

        Warning: This function is used for single element in trajectory, between to points, and for single axis only!
        '''

        n = degree + 1
        Q = np.zeros((n, n))
        
        for i in range(4, n):
            for j in range(4, n):
                c_i = math.factorial(i) / math.factorial(i - 4) 
                c_j = math.factorial(j) / math.factorial(j - 4) 
                Q[i, j] = 2 * c_i * c_j * (T ** (i + j - 7)) / (i + j - 7)
                
        return Q
    
    
    def single_segment_constrain(self, p0, pk):
   
        A_eq = np.zeros((8, 8))
        b_eq = np.zeros(8)

        # --- initial state (s = 0) ---
        A_eq[0, :] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # p(0) = p0
        A_eq[1, :] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # v(0) = 0
        A_eq[2, :] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # a(0) = 0
        A_eq[3, :] = [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0]  # j(0) = 0

        # --- final state (s = 1.0) ---
        A_eq[4, :] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # p(1) = pk
        A_eq[5, :] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # v(1) = 0
        A_eq[6, :] = [0.0, 0.0, 2.0, 6.0, 12.0, 20.0, 30.0, 42.0]  # a(1) = 0
        A_eq[7, :] = [0.0, 0.0, 0.0, 6.0, 24.0, 60.0, 120.0, 210.0] # j(1) = 0

        b_eq[0] = p0
        b_eq[4] = pk

        return A_eq, b_eq
    
    def find_trajectory(self):
        C = []
        time_vect = self.allocate_time(speed=self.speed)
        segments_n = time_vect.shape[0] 

        for n in range(segments_n):
            C_axis = []
            for axis in range(3):
                p0 = self.waypoints[n, axis]
                pk = self.waypoints[n + 1, axis]

                # get constrain for equation (A_eq * c = b_eq)
                A_eq, b_eq = self.single_segment_constrain(p0, pk)
                
                try:
                    c_opt = np.linalg.solve(A_eq, b_eq)
                except np.linalg.LinAlgError:
                    self.get_logger().error(f"[Trajectory Error] Cannot solve equation {n}")
                    c_opt = np.zeros(8)
                
                C_axis.append(c_opt)
            C.append(C_axis)
        
        C_np = np.array(C)
        global_time_vect = np.cumsum(time_vect)
        return C_np, global_time_vect, time_vect


    
    def sample_trajectory(self, c, local_t, T):
        # Zabezpieczenie przed podaniem T=0 (jeśli wygenerowano zerowy odcinek czasu)
        if T < 1e-4:
            return c[0], 0.0, 0.0

        s = local_t / T  # Czas znormalizowany od 0 do 1
        
        # 1. Pozycja
        p = c[0] + c[1]*s + c[2]*s**2 + c[3]*s**3 + c[4]*s**4 + c[5]*s**5 + c[6]*s**6 + c[7]*s**7
        
        # 2. Prędkość - wyliczona ze znormalizowanego czasu, podzielona przez T!
        v_s = c[1] + 2*c[2]*s + 3*c[3]*s**2 + 4*c[4]*s**3 + 5*c[5]*s**4 + 6*c[6]*s**5 + 7*c[7]*s**6
        v = v_s / T
        
        # 3. Przyspieszenie - podzielone przez T^2
        a_s = 2*c[2] + 6*c[3]*s + 12*c[4]*s**2 + 20*c[5]*s**3 + 30*c[6]*s**4 + 42*c[7]*s**5
        a = a_s / (T**2)

        return p, v, a
    
  
def main(args=None):
    rclpy.init(args=args)

    node = MinimumSnapTrajectory()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
