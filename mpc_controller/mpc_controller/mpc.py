import rclpy
from rclpy.node import Node

from scipy.spatial.transform import Rotation as r
import numpy as np
import math

from nav_msgs.msg._odometry import Odometry
from cf_control_msgs.msg import ThrustAndTorque

from drone_model_msgs.msg import PolynomialTrajectory

from trajectory.trajectory.state_from_traj import state_from_output

from acados_template import AcadosOcpSolver
from casadi import SX, vertcat, horzcat, mtimes, inv, cross

class lee_controller(Node):
    def __init__(self):
        super().__init__('LeeController')

        # --- controller configuration ---
        
        # --- drone model and parameters ---
        self.declare_parameter('m', 1.0)  # [kg]
        self.m = self.get_parameter('m').value

        self.declare_parameter('J', [0.0]*9) # [kg * m^2]
        J_flat = self.get_parameter('J').value
        self.J = np.array(J_flat, dtype = np.float32).reshape((3,3))
    
        self.g = 9.81
        
        # --- ROS publishers and subscribers ---
        self.declare_parameter('dt', 0.01) 
        self.dt = self.get_parameter('dt').value
        self.physics_timer = self.create_timer(self.dt, self._set_output)

        # drone state subsriber 
        self.declare_parameter('ROS2_topic_name_state', '/crazyflie/Odometry') 
        ROS2_topic_name_state = self.get_parameter('ROS2_topic_name_state').value

        self.subscriber_state = self.create_subscription(
            Odometry, ROS2_topic_name_state, self._get_input, 10
        )

        # drone trajectory subscriber
        self.declare_parameter('ROS2_topic_name_setpoint', '/crazyflie/Trajectory') 
        ROS2_topic_name_setpoint = self.get_parameter('ROS2_topic_name_setpoint').value

        self.subscriber_setpoint = self.create_subscription(
            PolynomialTrajectory, ROS2_topic_name_setpoint, self.setpoint, 10
        )

        # drone ThrustAndTorque publisher 
        self.declare_parameter('ROS2_topic_name_ThrustAndTorque', '/cf_control/control_command') 
        ROS2_topic_name_thrustandtorque = self.get_parameter('ROS2_topic_name_ThrustAndTorque').value

        self.publisher = self.create_publisher(ThrustAndTorque, ROS2_topic_name_thrustandtorque, 10)

        # --- Acados Solver --- 

        try:
            self.solver = AcadosOcpSolver(
                None, 
                generate=False, # Do not generate code again
                build=False,    # Do not compile again
                json_file='acados_ocp.json' # Path to generated code
            )
            self.get_logger().info("Solver loaded!")
        except Exception as e:
            self.get_logger().error(f"Error while loading solver: {e}")
            raise e
        
        # --- solver parameters  ---

        # parameters shall be the same as declared in generate_slver.py!!!

        # horizon size 
        self.declare_parameter('horizon_N', 20) 
        self.horizon_N = self.get_parameter('horizon_N').value

        # solver time interval 
        self.declare_parameter('horizon_T', 1.0) 
        self.horizon_T = self.get_parameter('horizon_T').value

        self.optim_dt = self.horizon_T / self.horizon_N

        # solver state 
        self.actual_state = np.zeros((13))
        self.actual_state[6] = 1.0 # quaternion init


    def setpoint(self, msg: PolynomialTrajectory):
    
        if msg is not None:
            self.poly_x = np.array(msg.polynomial_x)
            self.poly_y = np.array(msg.polynomial_y)
            self.poly_z = np.array(msg.polynomial_z)
            self.poly_T = msg.duration # duration of segment 
            
            # local time (from start of this segment)
            self.poly_t_local_start = msg.timestamp
            
            # get ROS timer state when msg recived 
            self.poly_start_time_ros = self.get_clock().now()

    def _sample_polynomial(self, c, s, T):
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

        # Jerk - 4th derivative
        j_s = 6*c[3] + 24*c[4]*s + 60*c[5]*s**2 + 120*c[6]*s**3 + 210*c[7]*s**4
        j = j_s / (T**3)

        # Snap - 5th derivative
        s_s = 24*c[4] + 120*c[5]*s + 360*c[6]*s**2 + 840*c[7]*s**3
        s = s_s / (T**4)
        return p, v, a, j, s


    def _get_input(self, msg: Odometry):
      
        if msg is not None:
            # pose - global 
            pos = msg.pose.pose.position
            
            # orientation - global 
            quat = msg.pose.pose.orientation
            self.act_rotation_q = np.array([quat.x, quat.y, quat.z, quat.w])
            R_actual = r.from_quat(self.act_rotation_q)
            
            # velocity - local 
            vel_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            act_linear_vel = R_actual.apply(vel_body) 
            
            # angular velocity - local 
            ang_vel = msg.twist.twist.angular

            self.actual_state = np.array([pos.x, pos.y, pos.z,
                                          quat.x, quat.y, quat.z, quat.w,
                                          act_linear_vel[0], act_linear_vel[1], act_linear_vel[2],
                                          ang_vel.x, ang_vel.y, ang_vel.z])


    def control_loop_iter(self):
        if self.poly_start_time_ros is not None:
            
            # --- get time stamp --- 

            # for sampling polynomial: 

            # max segmemnt duration
            T = max(self.poly_T, 1e-4) 

            # get normalized time start 
            now = self.get_clock().now()
            dt_since_msg = (now - self.poly_start_time_ros).nanoseconds / 1e9
            t_local = self.poly_t_local_start + dt_since_msg
            # s = t_local / T 
            
            # --- set actual state as initial condition --- 
            self.solver.set(0, "lbx", self.actual_state)
            self.solver.set(0, "ubx", self.actual_state)

            # --- set refernce for optimization (optimal trajectory) --- 

            for k in range(self.horizon_N):

                future_time = t_local + self.optim_dt * k 
                s_t = future_time / T

                yref = np.zeros(17) # init ref (state [13] + control [6])
                
                # get trajectory and it's derivative
                p_x, v_x, a_x, j_x, s_x = self._sample_polynomial(self.poly_x, s_t, self.horizon_T)
                p_y, v_y, a_y, j_y, s_y = self._sample_polynomial(self.poly_y, s_t, self.horizon_T)
                p_z, v_z, a_z, j_z, s_z = self._sample_polynomial(self.poly_z, s_t, self.horizon_T)

                x_ref = np.array([p_x, p_y, p_z, 0.0])
                dx_ref = np.array([v_x, v_y, v_z, 0.0])
                d2x_ref = np.array([a_x, a_y, a_z, 0.0])
                d3x_ref = np.array([j_x, j_y, j_z, 0.0])
                d4x_ref = np.array([s_x, s_y, s_z, 0.0])

                # get reference state from flat outputs 
                state_ref, control_ref = state_from_output(self.m, self.J, self.g, 
                                                        x_ref, dx_ref, d2x_ref, d3x_ref, d4x_ref)

                # set references 
                yref[0:13] = state_ref # reference state
                yref[13:17] = control_ref # reference control 

                self.solver.set(k, "yref", yref)

           # --- final state in optimization ---  
            s_t = (t_local + self.horizon_T) / T

            p_x, v_x, a_x, j_x, s_x = self._sample_polynomial(self.poly_x, s_t, self.horizon_T)
            p_y, v_y, a_y, j_y, s_y = self._sample_polynomial(self.poly_y, s_t, self.horizon_T)
            p_z, v_z, a_z, j_z, s_z = self._sample_polynomial(self.poly_z, s_t, self.horizon_T)

            x_ref = np.array([p_x, p_y, p_z, 0.0])
            dx_ref = np.array([v_x, v_y, v_z, 0.0])
            d2x_ref = np.array([a_x, a_y, a_z, 0.0])
            d3x_ref = np.array([j_x, j_y, j_z, 0.0])
            d4x_ref = np.array([s_x, s_y, s_z, 0.0])

            # get reference state from flat outputs 
            state_ref_end, _ = state_from_output(self.m, self.J, self.g, 
                                                    x_ref, dx_ref, d2x_ref, d3x_ref, d4x_ref)
            yref_e = state_ref_end

            self.solver.set(self.horizon_N, "yref", yref_e)

            # --- Solve optimization --- 
            status = self.solver.solve()
            
            if status != 0:
                self.get_logger().warning(f"Acados solver error: {status}")

            # get first control value 
            u_opt = self.solver.get(0, "u")
            thrust = u_opt[0]
            torque = u_opt[1:]
            return thrust, torque 
        else: 
            return 0.0, np.array([0.0, 0.0, 0.0])
        
    def _set_output(self):
        thrust, torque = self.control_loop_iter()
        # --- Publish ---
        msg = ThrustAndTorque()
        msg.collective_thrust = float(thrust)
        msg.torque.x = float(torque[0])
        msg.torque.y = float(torque[1])
        msg.torque.z = float(torque[2])
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = lee_controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
