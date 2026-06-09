from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, mtimes, inv, cross

class DroneModel:
    def __init__(sel2f):
        pass

    def setup_model(self):
        model = AcadosModel()
        model.name = "UAV_model"
        
        # define drone parameters
        mass = SX.sym('mass')
        g = 9.81

        Ixx = SX.sym('Ixx')
        Iyy = SX.sym('Iyy')
        Izz = SX.sym('Izz')

        Ixy = SX.sym('Ixy')
        Ixz = SX.sym('Ixz')
        Iyz = SX.sym('Iyz')

        J = vertcat(
            horzcat(Ixx, Ixy, Ixz),
            horzcat(Ixy, Iyy, Iyz),
            horzcat(Ixz, Iyz, Izz)
        )
        
        J_inv = inv(J)

        # define parameters vector
        p_params = vertcat(mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

        # symbolic vector - model state
        p = SX.sym('p', 3)          # pose [x, y, z]
        q = SX.sym('q', 4)          # orientation [qx, qy, qz, qw] 
        v = SX.sym('v', 3)          # linear vel [vx, vy, vz]
        omega = SX.sym('omega', 3)  # angular vel [wx, wy, wz]
        
        model.x = vertcat(p, q, v, omega)

        # symbolic vector - model state derivative
        dp = SX.sym('dp', 3)          # pose derivative
        dq = SX.sym('dq', 4)          # orientation derivative
        dv = SX.sym('dv', 3)          # linear vel derivative
        domega = SX.sym('domega', 3)  # angular vel derivative
        
        model.xdot = vertcat(dp, dq, dv, domega)

        # symbolic vector - control 
        thrust = SX.sym('thrust')
        torque = SX.sym('torque', 3) 

        model.u = vertcat(thrust, torque)

        # drone model equations (Explicit ODE)

        # derivative of pose dp = v
        x_dot = v 

        # derivative of linear velocity - dynamics equations
        thrust_vect_local = vertcat(0, 0, thrust)
    
        qx, qy, qz, qw = q[0], q[1], q[2], q[3] # extract quaternions
        local2global_rot = vertcat(
            horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)),
            horzcat(2*(qx*qy + qw*qz),   1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            horzcat(2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2))
        )

        # compose gravity vector
        gravity_vect_glob = vertcat(0, 0, -g)
        
        # zamienione na symboliczną zmienną mass
        v_dot = gravity_vect_glob + (mtimes(local2global_rot, thrust_vect_local) / mass)
  
        # quaternion dynamics 
        wx, wy, wz = omega[0], omega[1], omega[2]
        
        Q_w = vertcat(
            horzcat(qw, -qz,  qy,  qx),
            horzcat(qz,  qw, -qx,  qy),
            horzcat(-qy,  qx,  qw,  qz),
            horzcat(-qx, -qy, -qz,  qw)
        )
        omega_quat = vertcat(wx, wy, wz, 0.0) 
        
        # # CZYSTA KINEMATYKA (BEZ c_stab!)
        # q_dot = 0.5 * mtimes(Q_w, omega_quat)

   
        quat_norm_sq = qx**2 + qy**2 + qz**2 + qw**2
        c_stab = 50.0  # Silniejszy współczynnik ściągający na kulę jednostkową
        # Prawidłowa kinematyka + stabilizacja do sfery jednostkowej
        q_dot = 0.5 * mtimes(Q_w, omega_quat) + 0.5 * c_stab * q * (1.0 - quat_norm_sq)


        # angular acceleration - dynamics rotation equation
        Jw = mtimes(J, omega)              
        cross_prod = cross(omega, Jw)      
        omega_dot = mtimes(J_inv, (torque - cross_prod))

        # Compose dynamic equation matrix
        f_expl = vertcat(x_dot, q_dot, v_dot, omega_dot)
        
        model.f_expl_expr = f_expl 
        model.p = p_params

        return model
    