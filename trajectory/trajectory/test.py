import numpy as np 
# import scipy.spatial.transform.Rotation as R
from scipy.spatial.transform import Rotation as R
import pandas as pd


from .state_from_traj import state_from_output



def test(testcase_pth, details = True):

    # read from csv
    df = pd.read_csv(testcase_pth)
    
    for row in df.itertuples():
        testname = row.test_name
   
        x = np.stack([row.in_pos_x, row.in_pos_y, row.in_pos_z, row.in_yaw], axis=-1)
        dx = np.stack([row.in_vel_x, row.in_vel_y, row.in_vel_z, row.in_yaw_rate], axis=-1)
        d2x = np.stack([row.in_acc_x, row.in_acc_y, row.in_acc_z, row.in_yaw_acceleration], axis=-1)
        d3x = np.stack([row.in_jerk_x, row.in_jerk_y, row.in_jerk_z], axis=-1)
        d4x = np.stack([row.in_snap_x, row.in_snap_y, row.in_snap_z], axis=-1)
        m = row.in_mass
        g = row.in_gravity
        J = np.diag([row.in_I_xx, row.in_I_yy, row.in_I_zz])

        gt_state = np.array([
            row.out_pos_x, row.out_pos_y, row.out_pos_z, 
            row.out_quat_w, row.out_quat_x, row.out_quat_y, row.out_quat_z, 
            row.out_vel_x, row.out_vel_y, row.out_vel_z, 
            row.out_omega_x, row.out_omega_y, row.out_omega_z
        ])
        
        gt_control = np.array([row.out_thrust, row.out_torque_x, row.out_torque_y, row.out_torque_z])

        # ---
        y_state, y_control = state_from_output(m, J, g, x, dx, d2x, d3x, d4x)

        testcase_res_state = np.allclose(y_state, gt_state)
        testcase_res_control = np.allclose(y_control, gt_control)

        if testcase_res_state and testcase_res_control:
            testcase_res = 'PASSED' 
        else:
            testcase_res = 'FAILED'

        print('='*50,f'\nTestcase: {testname}: {testcase_res}\n')

        # --- results ---
        if details:
            print('Details:')
            print('> Position:')
            for i in range(gt_state.shape[0]): # POPRAWKA: shape to krotka, iterujemy po jej pierwszym wymiarze
                my, gt = y_state[i], gt_state[i]
                if np.allclose(my, gt):
                    res = 'PASSED'
                else: 
                    res = 'FAILED'
                print(f'    gt: {gt}, my: {my}, result: {res}')

            print('> Control:')
            for i in range(gt_control.shape[0]): # POPRAWKA j.w.
                my, gt = y_control[i], gt_control[i]
                if np.allclose(my, gt): 
                    res = 'PASSED'
                else: 
                    res = 'FAILED'
                print(f'    gt: {gt}, my: {my}, result: {res}')



testcase_pth = 'C:/Users/janis/OneDrive/Pulpit/Studia/Semestr_III/Aerial_Robotics/trajectory/trajectory_from_flat_output_test_data.csv'

test(testcase_pth, details = False)