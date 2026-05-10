
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
       
        df = pd.read_csv(self.waypoints_pth)
        # self.waypoints = df.to_numpy() 
        self.waypoints = np.loadtxt(self.waypoints_pth, delimiter=',', skiprows=1, dtype=float)
        
        self.declare_parameter('desire_speed', 1.0)
        self.speed = self.get_parameter('desire_speed').value

        # === ROS publishers and subscribers ===
        self.declare_parameter('dt', 0.01)
        self.dt = self.get_parameter('dt').value

        self.physics_timer = self.create_timer(self.dt, self.set_output)

        # drone ThrustAndTorque publisher 
        self.declare_parameter('ROS2_topic_name_TrajectorySetpoint', '/crazyflie/Setpoints') 
        ROS2_topic_name_TrajectorySetpoint = self.get_parameter('ROS2_topic_name_TrajectorySetpoint').value

        self.publisher = self.create_publisher(TrajectorySetpoint, ROS2_topic_name_TrajectorySetpoint, 10)

        # calculate trajectory 
        self.trajectory_coefficients, self.global_time_vect, self.local_time_vect  = self.find_trajectory()
        self.global_time_max = self.global_time_vect[-1]

    
        self.last_known_yaw = 0.0


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
        

        p_x, v_x, a_x = self.sample_trajectory(self.trajectory_coefficients[segment_idx, 0], local_t)
        p_y, v_y, a_y = self.sample_trajectory(self.trajectory_coefficients[segment_idx, 1], local_t)
        p_z, v_z, a_z = self.sample_trajectory(self.trajectory_coefficients[segment_idx, 2], local_t)


        # calculate yaw angle
        speed_sq = v_x**2 + v_y**2
        
        if speed_sq > 1e-4:
            yaw = math.atan2(v_y, v_x)
            yaw_rate = (v_x * a_y - v_y * a_x) / speed_sq
        else:
            yaw = self.last_known_yaw 
            yaw_rate = 0.0

        self.last_known_yaw = yaw

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
    
    
    def single_segment_constrain(self, t, p0, pk):
        '''
        Matrix equation: A_eq * c = b_eq

        For 7th degree polynomial, we need 8 constrains as we are looking for 8 unknown variables. 
        We define constrains as matrix equation:
        A_eq * c = b_eq => c = A_eq * b_eq
        A_eq is matrxi that defines coeefitiens, that after linear combination with c gives us equation, setting derivatives etc. 
        b_eq is our constrians values.
        It is needed to set manually cooefitiens to get n-th derivatives for t=0 or t=t_k. 

        Warning: This function is used for single element in trajectory, between to points, and for single axis only!
      
        '''


        A_eq = np.zeros((8, 8))
        b_eq = np.zeros(8)

        # --- set initial position (t=0) ---
        A_eq[0, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        b_eq[0] = p0

        # --- set initial velocity (t=0) ---
        A_eq[1, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        b_eq[1] = 0.0

        # --- set initial acceleration (t=0) ---
        A_eq[2, :] = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        b_eq[2] = 0.0

        # --- set initial jerk (t=0) -> 3rd derivative is 6*c3 ---
        A_eq[3, :] = np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0]) 
        b_eq[3] = 0.0

        # --- set final position (t=T) ---
        A_eq[4, :] = np.array([1.0, t, t**2, t**3, t**4, t**5, t**6, t**7]) 
        b_eq[4] = pk

        # --- set final velocity (t=T) ---
        A_eq[5, :] = np.array([0.0, 1.0, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]) 
        b_eq[5] = 0.0

        # --- set final acceleration (t=T) ---
        A_eq[6, :] = np.array([0.0, 0.0, 2.0, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5]) 
        b_eq[6] = 0.0

        # --- set final jerk (t=T) ---
        A_eq[7, :] = np.array([0.0, 0.0, 0.0, 6.0, 24*t, 60*t**2, 120*t**3, 210*t**4]) 
        b_eq[7] = 0.0

        return A_eq, b_eq

    
    def find_trajectory(self):

        C = []
        
        time_vect = self.allocate_time(speed = self.speed)
    
        segments_n = time_vect.shape[0] 

        for n in range(segments_n):
            
            
            t = time_vect[n]
            Q = self.compute_Q_matrix(t)
            
            C_axis = []
            for axis in range(3):
                p0 = self.waypoints[n, axis]
                pk = self.waypoints[n + 1, axis]

                A_eq, b_eq = self.single_segment_constrain(t, p0, pk)
                
                constraints = {'type': 'eq', 'fun': lambda c: np.dot(A_eq, c) - b_eq}
                c0 = np.zeros(Q.shape[0])
                
                def cost_function(c):
                    return np.dot(c.T, np.dot(Q, c))
                
                # Run solver
                result = minimize(cost_function, c0, method='SLSQP', constraints=constraints)
                
                C_axis.append(result.x)

            C.append(C_axis)

        
        C_np = np.array(C) # shape (segments_n, axis, coeeficient) (n, 3, 8)
        global_time_vect = np.cumsum(time_vect)

        return C_np, global_time_vect, time_vect


    
    def sample_trajectory(self, c, t):
        
        # 1. position
        p = (c[0] + 
             c[1] * t + 
             c[2] * t**2 + 
             c[3] * t**3 + 
             c[4] * t**4 + 
             c[5] * t**5 + 
             c[6] * t**6 + 
             c[7] * t**7)
        
        # 2. velocity - 1st derivative
        v = (c[1] + 
             2 * c[2] * t + 
             3 * c[3] * t**2 + 
             4 * c[4] * t**3 + 
             5 * c[5] * t**4 + 
             6 * c[6] * t**5 + 
             7 * c[7] * t**6)
        
        # 3. Acceleration - 2nd derivative
        a = (2 * c[2] + 
             6 * c[3] * t + 
             12 * c[4] * t**2 + 
             20 * c[5] * t**3 + 
             30 * c[6] * t**4 + 
             42 * c[7] * t**5)

        return p, v, a
    
  
def main(args=None):
    rclpy.init(args=args)

    node = MinimumSnapTrajectory()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
