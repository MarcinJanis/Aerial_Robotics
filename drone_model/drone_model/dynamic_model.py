import sys 

import numpy as np 
import scipy 
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from cf_control_msgs.msg import ThrustAndTorque
from drone_model_msgs.msg import DroneState


# def q_conjugate(q):
#     return q * np.array([-1, -1, -1, 1])

def hamilton_product(q1, q2):

    x1, y1, z1, w1 = q1

    x2, y2, z2, w2 = q2

    # vector (x, y, z) - complex components
    x =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y =  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z =  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # scalar (w) - real component
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return np.array([x, y, z, w])



class DroneDynamicModel(Node):

    def __init__(self, m, J, dt):
        super().__init__('DroneDynamicModel')

        # --- physical parameter ---
        self.g = np.array([0.0, 0.0, -9.81])

        # --- drone parameters ---
        self.m = m # [kg]

        self.J = J # np.array([[1.0, 0.0, 0.0],
                   #           [0.0, 1.0, 0.0], 
                   #           [0.0, 0.0, 1.0]]) 

        # --- system parameters ---
        self.dt = dt # [s] 
        self.n = 0

        # --- state --- 
        self.drone_state = np.zeros(13) # state
        self.drone_state[9] = 1.0 # normaloized quaterion

        # --- forces and torque --- 
        self.torque = np.array([0.0, 0.0, 0.0]) # [N]
        self.thrust = np.array([0.0, 0.0, 0.0]) # [Nm]

        self.subscriber = self.create_subscription(
            ThrustAndTorque,
            "/model/control",
            self.get_input,
            10 
        )

        self.publisher = self.create_publisher(
            DroneState, 
            "/model/State", 
            10)

        self.physics_timer = self.create_timer(self.dt, self.step)


    def get_input(self, msg: ThrustAndTorque):
        self.thrust = np.array([0.0, 0.0, msg.collective_thrust])
        self.torque = np.array([msg.torque.x, msg.torque.y, msg.torque.z])

    def get_state(self):
        translation = self.drone_state[0:3]
        linear_vel = self.drone_state[3:6]
        rotation = self.drone_state[6:10]
        angular_vel = self.drone_state[10:13]
        return translation, linear_vel, rotation, angular_vel

    def publish_output(self):
        msg = DroneState()

        translation, linear_vel, rotation, angular_vel = self.get_state()
        msg.timestamp = self.n*self.dt
        msg.translation = translation.tolist()
        msg.linear_vel = linear_vel.tolist()
        msg.rotation = rotation.tolist()
        msg.angular_vel = angular_vel.tolist()

        self.publisher.publish(msg)

    # ======================================================================
    # Dynamic of drone: x' = f(x)
    # State vector: x = [x, y, z, vx, vy, vz, qx, qy, qz, qw, dqx, dqy, dqz] 
    def f(self, x, torque, thrust):
        
        dx = np.zeros((13), dtype=np.float64)

        r = x[0:3] 
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        # linear movement dynamic: 
        # dr/dt = v
        dx[0:3] = v 

        # linear velocity dynamics: 
        # dv/dt = a = -g + T_body_frame = -g + q_glob2body * T_global_frame
        q_rot = R.from_quat(x[6:10]) # transformation object global frame -> body frame
        dx[3:6] = self.g + q_rot.apply(thrust)/self.m

        # quaterion dynamics: 
        omega_quat = np.concatenate([w/2, [0.0]], axis = 0)
        dx[6:10] = hamilton_product(q, omega_quat)

        # angular velocity dynamics:
        dx[10:13] = np.linalg.inv(self.J) @ (torque - np.cross(w, (self.J @ w)))

        return dx
    
    def step(self):
        # solve differential equation to get drone state
        self.n += 1

        # === rk4 ===
        k1 = self.f(self.drone_state, self.torque, self.thrust)
        k2 = self.f(self.drone_state + self.dt/2*k1, self.torque, self.thrust)
        k3 = self.f(self.drone_state + self.dt/2*k2, self.torque, self.thrust)
        k4 = self.f(self.drone_state + self.dt*k3, self.torque, self.thrust)
        
        state_next = self.drone_state + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        # nromalize quaterion 
        q = state_next[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm != 0:
            state_next[6:10] = q/q_norm

        self.drone_state = state_next

        self.publish_output()



def main(args=None):
    rclpy.init(args=args)

    # if len(sys.argv) > 1:
    #     visu_type = sys.argv[1]
    # else:
    #     visu_type = None

    # --- parameters --- 
    m = 1 # [kg]
    J = np.array([[1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    dt = 0.01 # [s]

    node = DroneDynamicModel(m, J, dt)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



# --- 
# to send msg via terminal:

# ros2 topic pub /model/control cf_control_msgs/msg/ThrustAndTorque "{timestamp: 0, collective_thrust: 9.81, torque: {x: 0.0, y: 0.0, z: 0.0}}"