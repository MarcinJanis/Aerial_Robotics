import cv2
import numpy as np
import rclpy
from acados_template import AcadosOcpSolver
from nav_msgs.msg._odometry import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as r
from trajectory.state_from_traj import state_from_output

from cf_control_msgs.msg import ThrustAndTorque
from drone_model_msgs.msg import PolynomialTrajectory


class mpc_controller(Node):
    def __init__(self):
        super().__init__('MPC_Controller')

        # --- controller configuration ---

        # --- drone model and parameters ---
        self.declare_parameter('m', 0.025)  # [kg]
        self.m = self.get_parameter('m').value

        self.declare_parameter('J', [0.0] * 9)  # [kg * m^2]
        J_flat = self.get_parameter('J').value
        self.J = np.array(J_flat, dtype=np.float32).reshape((3, 3))

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
        ROS2_topic_name_thrustandtorque = self.get_parameter(
            'ROS2_topic_name_ThrustAndTorque'
        ).value

        self.publisher = self.create_publisher(
            ThrustAndTorque, ROS2_topic_name_thrustandtorque, 10
        )

        # --- Acados Solver ---

        try:
            self.solver = AcadosOcpSolver(
                None,
                generate=False,  # Do not generate code again
                build=False,  # Do not compile again
                json_file='/home/developer/ros2_ws/src/mpc_controller/mpc_controller/acados_ocp.json',  # Path to generated code
            )
            self.get_logger().info('Solver loaded!')
        except Exception as e:
            self.get_logger().error(f'Error while loading solver: {e}')
            raise e

        # --- solver parameters  ---

        # parameters shall be the same as declared in generate_slver.py!!!

        # horizon size
        self.declare_parameter('horizon_N', 20)
        self.horizon_N = self.get_parameter('horizon_N').value

        # solver time interval
        self.declare_parameter('horizon_T', 0.4)
        self.horizon_T = self.get_parameter('horizon_T').value

        self.optim_dt = self.horizon_T / self.horizon_N

        # solver state
        self.actual_state = np.zeros((13))
        self.actual_state[6] = 1.0  # quaternion init

        self.poly_start_time_ros = None
        # self.poly_t_local_start = self.poly_start_time_ros
        # self.poly_T = self.horizon_T

        self.last_pos = None
        self.last_time = None
        self.filtered_vel = np.array([0.0, 0.0, 0.0])
        self.alpha = (
            0.3  # Współczynnik wygładzania (im mniejszy, tym bardziej gładkie, ale z opóźnieniem)
        )

    def setpoint(self, msg: PolynomialTrajectory):

        if msg is not None:
            self.poly_x = np.array(msg.polynomial_x)
            self.poly_y = np.array(msg.polynomial_y)
            self.poly_z = np.array(msg.polynomial_z)
            self.poly_T = msg.duration  # duration of segment

            # local time (from start of this segment)
            self.poly_t_local_start = msg.timestamp

            # get ROS timer state when msg recived
            self.poly_start_time_ros = self.get_clock().now()

    def _sample_polynomial(self, c, s, T):

        if T < 1e-4:
            return c[0], 0.0, 0.0, 0.0, 0.0

        if s >= 1.0:
            # Ewaluacja pozycji na samym końcu segmentu
            p = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7]
            # Wymuszamy zera dla pochodnych, ponieważ pozycja jest sztucznie zamrożona!
            return p, 0.0, 0.0, 0.0, 0.0
        elif s <= 0.0:
            return c[0], 0.0, 0.0, 0.0, 0.0

        # Pose - 0 derivative
        p = (
            c[0]
            + c[1] * s
            + c[2] * s**2
            + c[3] * s**3
            + c[4] * s**4
            + c[5] * s**5
            + c[6] * s**6
            + c[7] * s**7
        )

        # Velocity - 1st derivative
        v_s = (
            c[1]
            + 2 * c[2] * s
            + 3 * c[3] * s**2
            + 4 * c[4] * s**3
            + 5 * c[5] * s**4
            + 6 * c[6] * s**5
            + 7 * c[7] * s**6
        )
        v = v_s / T

        # Acceleration - 3rd derivative
        a_s = (
            2 * c[2]
            + 6 * c[3] * s
            + 12 * c[4] * s**2
            + 20 * c[5] * s**3
            + 30 * c[6] * s**4
            + 42 * c[7] * s**5
        )
        a = a_s / (T**2)

        # Jerk - 4th derivative
        j_s = 6 * c[3] + 24 * c[4] * s + 60 * c[5] * s**2 + 120 * c[6] * s**3 + 210 * c[7] * s**4
        j = j_s / (T**3)

        # Snap - 5th derivative
        s_s = 24 * c[4] + 120 * c[5] * s + 360 * c[6] * s**2 + 840 * c[7] * s**3
        s = s_s / (T**4)
        return p, v, a, j, s

    def _get_input2(self, msg: Odometry):
        if msg is not None:
            # 1. Niezawodna pozycja z Gazebo
            pos = msg.pose.pose.position
            current_pos = np.array([pos.x, pos.y, pos.z])

            # 2. Prawdziwy czas wyciągnięty z wiadomości symulatora
            current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # 3. Odporne na asynchronizację i "Jitter" obliczanie prędkości
            if hasattr(self, 'last_pos') and self.last_pos is not None:
                # Obliczamy PRAWDZIWE dt między paczkami
                dt = current_time - self.last_time

                # ZABEZPIECZENIE: Różniczkujemy tylko, jeśli upłynęło min. 5 ms.
                # Chroni to przed dzieleniem przez zera (Jitter) w Gazebo.
                if dt >= 0.005:
                    raw_vel = (current_pos - self.last_pos) / dt

                    # Wygładzamy prędkość filtrem EMA (alpha = 0.4)
                    alpha = 0.4
                    self.filtered_vel = alpha * raw_vel + (1.0 - alpha) * self.filtered_vel

                    # Zapisujemy historię TYLKO wtedy, gdy wykonaliśmy poprawne obliczenia
                    self.last_pos = current_pos
                    self.last_time = current_time
            else:
                self.filtered_vel = np.array([0.0, 0.0, 0.0])
                self.last_pos = current_pos
                self.last_time = current_time

            # 4. Orientacja i prędkość kątowa (surowe dane z żyroskopów Gazebo)
            quat = msg.pose.pose.orientation
            ang_vel = msg.twist.twist.angular

            # 5. Aktualizacja stanu drona DLA MPC
            self.actual_state = np.array(
                [
                    current_pos[0],
                    current_pos[1],
                    current_pos[2],
                    quat.x,
                    quat.y,
                    quat.z,
                    quat.w,
                    self.filtered_vel[0],
                    self.filtered_vel[1],
                    self.filtered_vel[2],
                    ang_vel.x,
                    ang_vel.y,
                    ang_vel.z,
                ]
            )

    def _get_input(self, msg: Odometry):

        if msg is not None:
            # pose - global
            pos = msg.pose.pose.position

            # orientation - global
            quat = msg.pose.pose.orientation
            self.act_rotation_q = np.array([quat.x, quat.y, quat.z, quat.w])
            R_actual = r.from_quat(self.act_rotation_q)

            # velocity - local
            vel_body = np.array(
                [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
            )
            act_linear_vel = vel_body

            # angular velocity - local
            ang_vel = msg.twist.twist.angular

            self.actual_state = np.array(
                [
                    pos.x,
                    pos.y,
                    pos.z,
                    quat.x,
                    quat.y,
                    quat.z,
                    quat.w,
                    act_linear_vel[0],
                    act_linear_vel[1],
                    act_linear_vel[2],
                    ang_vel.x,
                    ang_vel.y,
                    ang_vel.z,
                ]
            )

            # self.actual_state = np.array([0.0, 0.0, 0.2,   0.0, 0.0, 0.0, 1.0,   0.0, 0.0, 0.0,   0.0, 0.0, 0.0])

    def control_loop_iter(self, show=False):

        if show:
            ref_path = []
            opt_path = []

        if self.poly_start_time_ros is not None:
            # --- get time stamp ---
            # max segmemnt duration
            T = max(self.poly_T, 1e-4)
            self.get_logger().info(f'T: {T}')
            # get normalized time start
            now = self.get_clock().now()
            dt_since_msg = (now - self.poly_start_time_ros).nanoseconds / 1e9
            t_local = self.poly_t_local_start + dt_since_msg

            # --- set actual state as initial condition ---
            self.solver.set(0, 'lbx', self.actual_state)
            self.solver.set(0, 'ubx', self.actual_state)

            # --- set refernce for optimization (optimal trajectory) ---
            q_prev = self.actual_state[3:7].copy()

            for k in range(self.horizon_N):
                future_time = t_local + self.optim_dt * k
                s_t = future_time / T

                yref = np.zeros(17)  # init ref (state [13] + control [6])

                # get trajectory and it's derivative
                p_x, v_x, a_x, j_x, s_x = self._sample_polynomial(self.poly_x, s_t, T)
                p_y, v_y, a_y, j_y, s_y = self._sample_polynomial(self.poly_y, s_t, T)
                p_z, v_z, a_z, j_z, s_z = self._sample_polynomial(self.poly_z, s_t, T)

                v_sq = v_x**2 + v_y**2
                if v_sq < 1e-4:
                    yaw_ref = 0.0
                    dyaw_ref = 0.0
                    d2yaw_ref = 0.0
                else:
                    yaw_ref = np.atan2(v_y, v_x)
                    dyaw_ref = (v_x * a_y - v_y * a_x) / v_sq
                    d2yaw_ref = 0.0

                x_ref = np.array([p_x, p_y, p_z, yaw_ref])
                dx_ref = np.array([v_x, v_y, v_z, dyaw_ref])
                d2x_ref = np.array([a_x, a_y, a_z, d2yaw_ref])
                d3x_ref = np.array([j_x, j_y, j_z, 0.0])
                d4x_ref = np.array([s_x, s_y, s_z, 0.0])

                # get reference state from flat outputs
                state_ref, control_ref = state_from_output(
                    self.m, self.J, self.g, x_ref, dx_ref, d2x_ref, d3x_ref, d4x_ref
                )

                # Quaternion sign flip preventer
                if np.dot(state_ref[3:7], q_prev) < 0:
                    state_ref[3:7] = -state_ref[3:7]
                q_prev = state_ref[3:7].copy()

                # ===================================================
                if np.any(np.isnan(state_ref)):
                    self.get_logger().error(f'NaN in state_ref: {state_ref}')
                    return 0.0, np.array([0.0, 0.0, 0.0])
                if np.any(np.isnan(control_ref)):
                    self.get_logger().error(f'NaN in control_ref: {control_ref}')
                    return 0.0, np.array([0.0, 0.0, 0.0])
                # ====================================================
                # set references
                yref[0:13] = state_ref  # reference state
                yref[13:17] = control_ref  # reference control

                self.solver.set(k, 'yref', yref)

                self.solver.set(k, 'x', state_ref)
                self.solver.set(k, 'u', control_ref)

                if show:
                    ref_path.append([p_x, p_y, p_z])

            # --- final state in optimization ---
            s_t = (t_local + self.horizon_T) / T
            p_x, v_x, a_x, j_x, s_x = self._sample_polynomial(self.poly_x, s_t, T)
            p_y, v_y, a_y, j_y, s_y = self._sample_polynomial(self.poly_y, s_t, T)
            p_z, v_z, a_z, j_z, s_z = self._sample_polynomial(self.poly_z, s_t, T)

            v_sq = v_x**2 + v_y**2
            if v_sq < 1e-4:
                yaw_ref = 0.0
                dyaw_ref = 0.0
                d2yaw_ref = 0.0
            else:
                yaw_ref = np.atan2(v_y, v_x)
                dyaw_ref = (v_x * a_y - v_y * a_x) / v_sq
                d2yaw_ref = 0.0

            x_ref = np.array([p_x, p_y, p_z, yaw_ref])
            dx_ref = np.array([v_x, v_y, v_z, dyaw_ref])
            d2x_ref = np.array([a_x, a_y, a_z, d2yaw_ref])
            d3x_ref = np.array([j_x, j_y, j_z, 0.0])
            d4x_ref = np.array([s_x, s_y, s_z, 0.0])

            # get reference state from flat outputs
            state_ref_end, control_ref_end = state_from_output(
                self.m, self.J, self.g, x_ref, dx_ref, d2x_ref, d3x_ref, d4x_ref
            )
            yref_e = state_ref_end

            self.solver.set(self.horizon_N, 'yref', yref_e)

            # =============================================
            if np.any(np.isnan(self.actual_state)):
                self.get_logger().error(f'NaN in initial state: {self.actual_state}')
                return 0.0, np.array([0.0, 0.0, 0.0])
            # =============================================

            self.solver.set(self.horizon_N, 'yref', yref_e)
            self.solver.set(self.horizon_N, 'x', state_ref_end)

            # --- Solve optimization ---
            status = self.solver.solve()

            if status != 0:
                self.get_logger().warning(f'Acados solver error: {status}')

            if show:
                for i in range(self.horizon_N + 1):
                    x_opt = self.solver.get(i, 'x')
                    opt_path.append([x_opt[0], x_opt[1], x_opt[2]])
                    self._draw_cv2_radar(ref_path, opt_path)

            # get first control value
            u_opt = self.solver.get(0, 'u')
            thrust = u_opt[0]
            torque = u_opt[1:]
            return thrust, torque
        else:
            return 0.0, np.array([0.0, 0.0, 0.0])

    def control_loop_soft_start(self):

        if not hasattr(self, 'soft_start_time') or self.soft_start_time is None:
            self.soft_start_time = self.get_clock().now()
            self.start_x = self.actual_state[0]
            self.start_y = self.actual_state[1]
            self.start_z = self.actual_state[2]

            self.target_z = self.start_z + 0.2
            self.T_soft = 5.0

        now = self.get_clock().now()
        t_elapsed = (now - self.soft_start_time).nanoseconds / 1e9

        def get_soft_start_ref(t):
            if t >= self.T_soft:
                return self.target_z, 0.0, 0.0, 0.0, 0.0

            s = t / self.T_soft
            dz = self.target_z - self.start_z

            p = self.start_z + dz * (10 * s**3 - 15 * s**4 + 6 * s**5)
            v = (dz / self.T_soft) * (30 * s**2 - 60 * s**3 + 30 * s**4)
            a = (dz / self.T_soft**2) * (60 * s - 180 * s**2 + 120 * s**3)
            j = (dz / self.T_soft**3) * (60 - 360 * s + 360 * s**2)
            snap = (dz / self.T_soft**4) * (-360 + 720 * s)

            return p, v, a, j, snap

        # --- set actual state as initial condition ---
        self.solver.set(0, 'lbx', self.actual_state)
        self.solver.set(0, 'ubx', self.actual_state)

        # --- set reference for optimization ---
        q_prev = self.actual_state[3:7].copy()

        for k in range(self.horizon_N):
            yref = np.zeros(17)  # init ref (state [13] + control [6])

            # Przewidywany czas dla danego węzła horyzontu
            t_future = t_elapsed + k * self.optim_dt

            # Pobranie gładkiej referencji Z dla przyszłego czasu
            p_z, v_z, a_z, j_z, s_z = get_soft_start_ref(t_future)

            # Blokada pozycji X i Y z momentu inicjalizacji startu
            p_x, p_y = self.start_x, self.start_y
            v_x, v_y, a_x, a_y = 0.0, 0.0, 0.0, 0.0
            j_x, j_y, s_x, s_y = 0.0, 0.0, 0.0, 0.0

            v_sq = v_x**2 + v_y**2
            if v_sq < 1e-4:
                yaw_ref = 0.0
                dyaw_ref = 0.0
                d2yaw_ref = 0.0
            else:
                yaw_ref = np.atan2(v_y, v_x)
                dyaw_ref = (v_x * a_y - v_y * a_x) / v_sq
                d2yaw_ref = 0.0

            x_ref = np.array([p_x, p_y, p_z, yaw_ref])
            dx_ref = np.array([v_x, v_y, v_z, dyaw_ref])
            d2x_ref = np.array([a_x, a_y, a_z, d2yaw_ref])
            d3x_ref = np.array([j_x, j_y, j_z, 0.0])
            d4x_ref = np.array([s_x, s_y, s_z, 0.0])

            # get reference state from flat outputs
            state_ref, control_ref = state_from_output(
                self.m, self.J, self.g, x_ref, dx_ref, d2x_ref, d3x_ref, d4x_ref
            )

            # Quaternion sign flip preventer
            if np.dot(state_ref[3:7], q_prev) < 0:
                state_ref[3:7] = -state_ref[3:7]
            q_prev = state_ref[3:7].copy()

            # ===================================================
            if np.any(np.isnan(state_ref)):
                self.get_logger().error(f'NaN in state_ref: {state_ref}')
                return 0.0, np.array([0.0, 0.0, 0.0])
            if np.any(np.isnan(control_ref)):
                self.get_logger().error(f'NaN in control_ref: {control_ref}')
                return 0.0, np.array([0.0, 0.0, 0.0])
            # ====================================================

            # set references

            yref[0:13] = state_ref  # reference state

            # Preventer 1: Prevent from zero-norm quaternion
            if np.linalg.norm(yref[3:7]) < 0.1: 
                yref[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

            yref[13:17] = control_ref  # reference control

            self.solver.set(k, 'yref', yref)
            self.solver.set(k, 'x', state_ref)
            self.solver.set(k, 'u', control_ref)

        # --- final state in optimization ---
        t_end = t_elapsed + self.horizon_T
        p_z, v_z, a_z, j_z, s_z = get_soft_start_ref(t_end)

        p_x, p_y = self.start_x, self.start_y
        v_x, v_y, a_x, a_y = 0.0, 0.0, 0.0, 0.0
        j_x, j_y, s_x, s_y = 0.0, 0.0, 0.0, 0.0

        yaw_ref, dyaw_ref, d2yaw_ref = 0.0, 0.0, 0.0

        x_ref = np.array([p_x, p_y, p_z, yaw_ref])
        dx_ref = np.array([v_x, v_y, v_z, dyaw_ref])
        d2x_ref = np.array([a_x, a_y, a_z, d2yaw_ref])
        d3x_ref = np.array([j_x, j_y, j_z, 0.0])
        d4x_ref = np.array([s_x, s_y, s_z, 0.0])

        # get reference state from flat outputs
        state_ref_end, control_ref_end = state_from_output(
            self.m, self.J, self.g, x_ref, dx_ref, d2x_ref, d3x_ref, d4x_ref
        )
        yref_e = state_ref_end


        # Preventer 2: Prevent from zero-norm quaternion in reference
        if np.linalg.norm(yref_e[3:7]) < 0.1:
            yref_e[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

        # Preventer 3: refernec thrust is equal to hover thrust 
        if yref[13] < 0.01:
            yref[13] = self.m * self.g

        # =============================================
        if np.any(np.isnan(self.actual_state)):
            self.get_logger().error(f'NaN in initial state: {self.actual_state}')
            return 0.0, np.array([0.0, 0.0, 0.0])
        # =============================================

        self.solver.set(self.horizon_N, 'yref', yref_e)
        self.solver.set(self.horizon_N, 'x', state_ref_end)

        # --- Solve optimization ---
        status = self.solver.solve()

        if status != 0:
            self.get_logger().warning(f'Acados solver error: {status}')

        # get first control value
        u_opt = self.solver.get(0, 'u')
        thrust = u_opt[0]
        torque = u_opt[1:]
        return thrust, torque

    def _set_output(self):

        if self.poly_start_time_ros is None:
            thrust, torque = 0.0, np.array([0.0, 0.0, 0.0])  # self.control_loop_soft_start()
        else:
            thrust, torque = self.control_loop_iter(show=True)

        # --- Publish ---
        msg = ThrustAndTorque()
        msg.collective_thrust = float(thrust)
        msg.torque.x = float(torque[0])
        msg.torque.y = float(torque[1])
        msg.torque.z = float(torque[2])
        self.publisher.publish(msg)

        self.get_logger().info(f'\nthrust: {thrust}\ntorque: {torque}')

        self.get_logger().info(
            f'Pose:\n[x, y, z]: [{self.actual_state[0]:.3} ,{self.actual_state[1]:.3}, {self.actual_state[2]:.3}]'
            # f'Pose:\n[x, y, z]: [{self.actual_state[0]:.3} ,{self.actual_state[1]:.3}, {self.actual_state[2]:.3}]'
        )
        self.get_logger().info(
            f'Vel:\n[vx, vy, vz]: [{self.actual_state[7]:.3} ,{self.actual_state[8]:.3}, {self.actual_state[9]:.3}]'
        )

    def _draw_cv2_radar(self, ref_path, opt_path):
        # Tworzymy czarne tło o wymiarach 500x500 pikseli
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        cx, cy = 250, 250  # Środek ekranu
        scale = 150.0  # Skala: 1 metr to 150 pikseli na ekranie

        # Funkcja pomocnicza: przelicza współrzędne globalne na piksele względem drona
        def to_img(x, y):
            dx = x - self.actual_state[0]
            dy = y - self.actual_state[1]
            img_x = int(cx - dy * scale)  # Y drona to oś X na ekranie (lewo/prawo)
            img_y = int(cy - dx * scale)  # X drona to oś Y na ekranie (przód/tył)
            return (img_x, img_y)

        # 1. Rysowanie trajektorii referencyjnej (ZIELONA)
        for i in range(len(ref_path) - 1):
            pt1 = to_img(ref_path[i][0], ref_path[i][1])
            pt2 = to_img(ref_path[i + 1][0], ref_path[i + 1][1])
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            cv2.circle(img, pt1, 3, (0, 255, 0), -1)

        # 2. Rysowanie trajektorii zoptymalizowanej (CZERWONA)
        for i in range(len(opt_path) - 1):
            pt1 = to_img(opt_path[i][0], opt_path[i][1])
            pt2 = to_img(opt_path[i + 1][0], opt_path[i + 1][1])
            cv2.line(img, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(img, pt1, 3, (0, 0, 255), -1)

        # 3. Rysowanie samego drona (NIEBIESKIE KÓŁKO W ŚRODKU)
        drone_pt = to_img(self.actual_state[0], self.actual_state[1])
        cv2.circle(img, drone_pt, 8, (255, 0, 0), -1)

        # Dodanie siatki odniesienia (krzyż)
        cv2.line(img, (cx, 0), (cx, 500), (50, 50, 50), 1)
        cv2.line(img, (0, cy), (500, cy), (50, 50, 50), 1)

        # Zapis do pliku zamiast wyświetlania w oknie GUI
        self.frame_counter = getattr(self, 'frame_counter', 0) + 1

        # Pętla działa na 100Hz, więc 100 klatek = 1 sekunda
        if self.frame_counter % 100 == 0:
            # Pamiętaj o absolutnej ścieżce, żeby łatwo znaleźć plik!
            save_path = f'/home/developer/ros2_ws/src/mpc_controller/tmp_debug_img/mpc_radar_{self.frame_counter}.jpg'
            cv2.imwrite(save_path, img)


def main(args=None):
    rclpy.init(args=args)
    node = mpc_controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
