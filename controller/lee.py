import numpy as np
from rclpy.node import Node
from scipy.spatial.transform import Rotation as r

from cf_control_msgs.msg import ThrustAndTorque
from drone_model_msgs.msg import DroneState


class lee_controller(Node):
    def __init__(self, kx, kv, kR, kw, m, J, dt):
        super().__init__('LeeController')

        self.kx = kx
        self.kv = kv
        self.kR = kR
        self.kw = kw

        # --- physical parameter ---
        self.g = np.array([0.0, 0.0, -9.81])

        # --- drone parameters ---
        self.m = m # [kg]

        self.J = J # np.array([[1.0, 0.0, 0.0],
                   #           [0.0, 1.0, 0.0], 
                   #           [0.0, 0.0, 1.0]]) [kg * m^2]

        self.g = 9.81

        # === init dron state ===
        self.act_timestap = 0.0
        self.act_translation = np.zeros((3,))
        self.act_linear_vel = np.zeros((3,))
        self.act_rotation_q = np.zeros((4,))
        self.act_angular_vel = np.zeros((3,))

        # === ROS publishers and subscribers === 
        self.dt = dt

        self.subscriber = self.create_subscription(
            DroneState, 
            "/model/State", 
            self.get_input,
            10 
        )

        self.publisher = self.create_publisher(
            ThrustAndTorque,
            "/model/control",
            10)

        self.physics_timer = self.create_timer(self.dt, self.set_output)


    def setpoint(self, x, v, a, y, w, dw):
        '''
        input: 
        x - target postion, shape [3, ]
        v - target linear velocity, shape [3, ]
        y - target y angle, shape  []
        w - target angular velocity, shape [3, ]
        '''
        self.x_sp = x # displacement, [3,]
        self.v_sp = v # linear velocity, [3,], set to 0 (?)
        self.a_sp = a # linear acceleration [3,], set to 0 (?)
        self.y_sp = y # yaw, [1,]
        self.w_sp = w # yaw velocity, set to 0 (?) 
        self.dw_sp = dw # angular acceleration, [3, ], set to 0 (?)

    def output(self, x, v, q, w):
        
        '''
        input: 
        x - actual postion, shape [3, ]
        v - actual linear velocity, shape [3, ]
        q - actual rotation (norm quaternion), shape [4, ]
        w - actual angular velocity, shape [3, ]


        output: 
        '''
        ax_z_g = np.array([0, 0, 1]) # z axis in global ref frame 

        # === error: position and linear velocity ===
        ex = x - self.x_sp # position error [3,]
        ev = v - self.v_sp # velocity error [3, ]

        # === compose target rotation matrix === 
        
        # z axis of body frame in global reference frame
        T_target = -self.kx*ex -self.kv * ev -ax_z_g*self.m*self.g + self.m*self.a_sp # desire thrust vector
        if np.linalg.norm(T_target)<1e-4:
            b3 = - T_target / np.linalg.norm(T_target) # z axis of body frame 
        else: 
            b3 = ax_z_g #????

        # x axis of body frame in global reference frame - set by yaw setpoint 
        b1 = np.array([np.cos(self.y_sp), np.sin(self.y_sp), 0]) # x axis of body frame

        # y axis of body frame in global reference frame - forced by x and z axis 
        b2 = np.cross(b3, b1) # y axis of body frame
        b2 = b2 / np.linalg.norm(b2)

        # correction for b1 axis - make sure, that b1 is perpendicular to b2 and b3
        b1 = np.cross(b2, b3) # ???

        # compose
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
        ew = w - (R_actual.inv() * R_target * self.w_sp)

        # === compute control output ===

        # required forces in global frame transformed to local frame, 
        # and casted on vertical axis, shape [3, ]
        F = np.dot(T_target, (R_actual * ax_z_g))

        # required torque, [3, ]
        M = - self.kR * eR - self.kw * ew + np.cross(w, (self.J @ w)) - \
        self.J @ (hat(w) @ R_actual.inv() @ R_target @ self.w_sp - R_actual.inv() @ R_target @ self.dw_sp)

        return F, M 
    

    def get_input(self, msg:DroneState):

        if not msg is None: 
            self.act_timestap = msg.timestamp 
            self.act_translation = msg.translation
            self.act_linear_vel = msg.linear_vel
            self.act_rotation_q = msg.rotation
            self.act_angular_vel = msg.angular_vel


    def set_output(self):
        msg = ThrustAndTorque()

        thrust, torque = self.output(self.act_translation,
                                    self.act_linear_vel,
                                    self.act_rotation_q,
                                    self.act_angular_vel)

        msg.collective_thrust = thrust
        msg.torque = torque
        self.publisher.publish(msg)


    
def vee_map(x):
    '''
    skew-symmetric -> column vector
    '''
    return np.array([x[2, 1], x[0, 2], x[1, 0]])

def hat(x):
    '''
    columnt vector -> skew-symmetric
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])