import rclpy
from rclpy.node import Node

import numpy as np
import math 

# Zakładam, że nazwa wiadomości to PolynomialTrajectory w drone_model_msgs
from drone_model_msgs.msg import PolynomialTrajectory


class MinimumSnapTrajectory(Node):
    def __init__(self):
        super().__init__('MinSnapTraj')

        self.start_time = None

        # --- trajectory waypoints --- 
        self.declare_parameter('waypoints_pth', '/')
        self.waypoints_pth = self.get_parameter('waypoints_pth').value
       
        self.waypoints = np.loadtxt(self.waypoints_pth, delimiter=',', skiprows=1, dtype=float)
        
        self.declare_parameter('desire_speed', 1.0)
        self.speed = self.get_parameter('desire_speed').value

        # === ROS publishers and subscribers ===
        self.declare_parameter('dt', 0.05) 
        self.dt = self.get_parameter('dt').value

        # drone Trajectory publisher 
        self.declare_parameter('ROS2_topic_name_Trajectory', '/crazyflie/Trajectory') 
        ROS2_topic_name_Trajectory = self.get_parameter('ROS2_topic_name_Trajectory').value

        self.publisher = self.create_publisher(PolynomialTrajectory, ROS2_topic_name_Trajectory, 10)

        # Wyznaczenie płynnej trajektorii globalnej metodą Minimum Snap
        self.trajectory_coefficients, self.global_time_vect, self.local_time_vect = self.find_trajectory()
        self.global_time_max = self.global_time_vect[-1]

        self.physics_timer = self.create_timer(self.dt, self.set_output)


    def set_output(self):
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        
        now = self.get_clock().now()
        global_t = (now - self.start_time).nanoseconds / 1e9

        # Wybór odpowiedniego segmentu trajektorii
        if global_t >= self.global_time_max:
            segment_idx = len(self.local_time_vect) - 1
            local_t = float(self.local_time_vect[-1])
        else:
            segment_idx = np.searchsorted(self.global_time_vect, global_t)
            if segment_idx == 0:
                local_t = global_t
            else:
                # Czas globalny minus czas, w którym zakończył się poprzedni segment
                local_t = global_t - self.global_time_vect[segment_idx - 1]

        # Pobranie współczynników dla aktualnego segmentu
        c_x = self.trajectory_coefficients[segment_idx, 0].tolist()
        c_y = self.trajectory_coefficients[segment_idx, 1].tolist()
        c_z = self.trajectory_coefficients[segment_idx, 2].tolist()
        
        # Pobranie czasu trwania obecnego segmentu (T)
        segment_duration = float(self.local_time_vect[segment_idx])

        # Budowanie wiadomości ROS2
        msg = PolynomialTrajectory()
        
        msg.timestamp = float(local_t)
        msg.polynomial_x = c_x
        msg.polynomial_y = c_y
        msg.polynomial_z = c_z
        msg.duration = segment_duration  # Dodano długość trwania segmentu

        self.get_logger().info(f'Seg: {segment_idx}, T_seg: {segment_duration:.2f}s, Global t: {global_t:.2f}s')
        self.publisher.publish(msg)
 

    def allocate_time(self, speed=1.0):
        '''
        Wyznaczenie czasu trwania każdego segmentu na podstawie odległości między punktami
        '''
        dist = np.linalg.norm(self.waypoints[1:, :] - self.waypoints[:-1, :], axis=-1)
        time_vect = dist / speed
        # Zabezpieczenie przed ekstremalnie małymi wartościami czasu
        time_vect = np.clip(time_vect, 1e-3, None)
        return time_vect

    
    def compute_Q_matrix(self, T, degree=7):
        '''
        Oblicza macierz Hessianu Q dla funkcji kosztu Minimum Snap 
        w czasie znormalizowanym s w [0, 1].
        '''
        n = degree + 1
        Q = np.zeros((n, n))
        T = max(T, 1e-4) # Zabezpieczenie przed dzieleniem przez zero
        
        for i in range(4, n):
            for j in range(4, n):
                c_i = math.factorial(i) / math.factorial(i - 4) 
                c_j = math.factorial(j) / math.factorial(j - 4) 
                Q[i, j] = c_i * c_j / (i + j - 7) / (T ** 7)
                
        return Q
    
    
    def find_trajectory(self):
        '''
        Globalny optymalizator Minimum Snap rozwiązujący macierz KKT 
        w celu zapewnienia idealnej ciągłości pochodnych na punktach wspólnych.
        '''
        time_vect = self.allocate_time(speed=self.speed)
        M = time_vect.shape[0]  # Liczba segmentów
        
        # Macierz przechowująca współczynniki: [Liczba_segmentów, 3_osie, 8_współczynników]
        C_all = np.zeros((M, 3, 8))
        
        # 1. Konstrukcja globalnej macierzy kosztów Q (wspólna dla wszystkich osi)
        Q_global = np.zeros((8 * M, 8 * M))
        for k in range(M):
            Q_seg = self.compute_Q_matrix(time_vect[k])
            Q_global[8*k : 8*k+8, 8*k : 8*k+8] = 2.0 * Q_seg
            
        # 2. Budowanie ograniczeń i rozwiązywanie dla każdej osi osobno (X, Y, Z)
        for axis in range(3):
            A_list = []
            b_list = []
            
            # --- OGRANICZENIA POZYCJI W WAYPOINTACH ---
            for k in range(M):
                # Początek segmentu (s=0) musi być w punkcie k
                r_start = np.zeros(8 * M)
                r_start[8 * k] = 1.0
                A_list.append(r_start)
                b_list.append(self.waypoints[k, axis])
                
                # Koniec segmentu (s=1) musi być w punkcie k+1
                r_end = np.zeros(8 * M)
                r_end[8*k : 8*k+8] = 1.0
                A_list.append(r_end)
                b_list.append(self.waypoints[k+1, axis])
                
            # --- WARUNKI BRZEGOWE CAŁEJ TRAJEKTORII (Start i Meta) ---
            # Start (segment 0, s=0): v=0, a=0, j=0
            for i in [1, 2, 3]:
                r_b0 = np.zeros(8 * M)
                r_b0[i] = 1.0
                A_list.append(r_b0)
                b_list.append(0.0)
                
            # Koniec (ostatni segment M-1, s=1): v=0, a=0, j=0
            T_end = time_vect[-1]
            # Prędkość końcowa = 0
            r_ve = np.zeros(8 * M)
            for i in range(1, 8): r_ve[8*(M-1) + i] = i / T_end
            A_list.append(r_ve); b_list.append(0.0)
            # Przyspieszenie końcowe = 0
            r_ae = np.zeros(8 * M)
            for i in range(2, 8): r_ae[8*(M-1) + i] = i * (i-1) / (T_end**2)
            A_list.append(r_ae); b_list.append(0.0)
            # Szarpnięcie (jerk) końcowe = 0
            r_je = np.zeros(8 * M)
            for i in range(3, 8): r_je[8*(M-1) + i] = i * (i-1) * (i-2) / (T_end**3)
            A_list.append(r_je); b_list.append(0.0)
            
            # --- WARUNKI CIĄGŁOŚCI W PUNKTACH POŚREDNICH ---
            for k in range(M - 1):
                T_k = time_vect[k]
                T_k1 = time_vect[k+1]
                
                # Ciągłość prędkości: v_k(1) = v_k+1(0)
                r_vc = np.zeros(8 * M)
                for i in range(1, 8): r_vc[8 * k + i] = i / T_k
                r_vc[8 * (k+1) + 1] = -1.0 / T_k1
                A_list.append(r_vc); b_list.append(0.0)
                
                # Ciągłość przyspieszenia: a_k(1) = a_k+1(0)
                r_ac = np.zeros(8 * M)
                for i in range(2, 8): r_ac[8 * k + i] = i * (i-1) / (T_k**2)
                r_ac[8 * (k+1) + 2] = -2.0 / (T_k1**2)
                A_list.append(r_ac); b_list.append(0.0)
                
                # Ciągłość szarpnięcia (jerk): j_k(1) = j_k+1(0)
                r_jc = np.zeros(8 * M)
                for i in range(3, 8): r_jc[8 * k + i] = i * (i-1) * (i-2) / (T_k**3)
                r_jc[8 * (k+1) + 3] = -6.0 / (T_k1**3)
                A_list.append(r_jc); b_list.append(0.0)
                
                # Ciągłość snapu: s_k(1) = s_k+1(0)
                r_sc = np.zeros(8 * M)
                for i in range(4, 8): r_sc[8 * k + i] = i * (i-1) * (i-2) * (i-3) / (T_k**4)
                r_sc[8 * (k+1) + 4] = -24.0 / (T_k1**4)
                A_list.append(r_sc); b_list.append(0.0)
                
            A_eq = np.array(A_list)
            b_eq = np.array(b_list)
            
            # --- BUDOWANIE I ROZWIĄZYWANIE UKŁADU KKT ---
            num_constraints = A_eq.shape[0]
            KKT_A = np.zeros((8*M + num_constraints, 8*M + num_constraints))
            KKT_b = np.zeros(8*M + num_constraints)
            
            KKT_A[0:8*M, 0:8*M] = Q_global
            KKT_A[0:8*M, 8*M:] = A_eq.T
            KKT_A[8*M:, 0:8*M] = A_eq
            KKT_b[8*M:] = b_eq
            
            try:
                sol = np.linalg.solve(KKT_A, KKT_b)
                c_opt = sol[0 : 8*M]
            except np.linalg.LinAlgError:
                self.get_logger().error(f"[Trajectory Error] Nie można rozwiązać układu KKT dla osi {axis}")
                c_opt = np.zeros(8 * M)
                
            # Przekształcenie płaskiego wektora wyników z powrotem do macierzy segmentów
            C_all[:, axis, :] = c_opt.reshape((M, 8))
            
        global_time_vect = np.cumsum(time_vect)
        return C_all, global_time_vect, time_vect
  

def main(args=None):
    rclpy.init(args=args)
    node = MinimumSnapTrajectory()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()