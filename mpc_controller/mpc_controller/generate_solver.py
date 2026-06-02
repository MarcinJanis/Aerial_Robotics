import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as cs


from drone_model import DroneModel

def generate_mpc_solver():

    # --- Solver model and parameters --- 

    drone = DroneModel()
    model = drone.setup_model()
    
    ocp = AcadosOcp()
    ocp.model = model
    
    nx = model.x.size()[0]  # state vector size (13)
    nu = model.u.size()[0]  # control vector size (4)
    ny = nx + nu            # total optimize vector size (17)
    ny_e = nx               # size of target on the end of optimization (state only, control is nt relevant) (13)
    
    N = 20                  # Prediction horizon
    Tf = 1.0                # Prediction size: (dt = Tf/N)
    
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    ocp.code_export_directory = 'c_generated_code' # directory for generated c files

    # Define cost funciton
    # Non-linear least squares
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    
    # Mówimy solverowi, co dokładnie ma minimalizować. 
    # Chcemy minimalizować błąd między (stanami i sterowaniami) a referencją.
    model.cost_y_expr = cs.vertcat(model.x, model.u)
    model.cost_y_expr_e = model.x
    
    # Weights matrix for state 
    Q = np.diag([
        50.0, 50.0, 50.0,   # px, py, pz 
        1.0, 1.0, 1.0, 1.0, # qx, qy, qz, qw 
        5.0, 5.0, 5.0,      # vx, vy, vz 
        0.1, 0.1, 0.1       # wx, wy, wz 
    ])
    
    # Weights for control
    R = np.diag([
        0.1,                # thrust cost (punishment for using proppelers)
        1.0, 1.0, 1.0       # torques (punishment for to big torques - smoother trajectory)
    ])
    
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q        # On the end of optimization, only state is consider, control is nor relevant 
    
    # Init cost functions with zeros 
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # --- Solver constrains ---
    # [thrust, torque_x, torque_y, torque_z]
    ocp.constraints.lbu = np.array([0.0, -0.015, -0.015, -0.015])  # lower bounds
    ocp.constraints.ubu = np.array([0.8, 0.015, 0.015, 0.015])  # upper bounds
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])  # idxs
    
    # --- initial state (init with zeros) ---
    x0 = np.zeros(nx)
    x0[6] = 1.0  # Set quaternion to norm quat 
    ocp.constraints.x0 = x0

    # --- Solver configuration  --- 
    # Initial values of online parametesr 
    # [mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    ocp.parameter_values = np.array([0.025, 
                                     16.571710e-06, 16.655602e-06, 29.261652e-06, 
                                     0.830806e-06, 0.718277e-06, 1.800197e-06])
    
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  
    
    ocp.solver_options.sim_method_num_stages = 4  # RK4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver_iter_max = 50
    
    # --- Generate code --- 
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    print("Solver code generated succesfully!")

if __name__ == "__main__":
    generate_mpc_solver()

    