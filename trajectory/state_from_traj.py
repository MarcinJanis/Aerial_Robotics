# extract drone state [x, y, z]  
# from flat output [x, y, z, psi].

import numpy as np 
# import scipy.spatial.transform.Rotation as R
from scipy.spatial.transform import Rotation as R
import pandas as pd



def state_from_output(m, J, g, input, dinput, d2input, d3input, d4input):

    x = input[:3]  # position
    dx = dinput[:3]  # derivative of position
    d2x = d2input[:3]  # second derivative of position
    d3x = d3input[:3]  # third derivative of position
    d4x = d4input[:3]

    psi = input[-1]  # yaw angle
    dpsi = dinput[-1]  # derivative of yaw 
    d2psi = d2input[-1]

    # parameters 
    zw = np.array([0, 0, 1])  # z-axis, global frame

    # --- thrust --- 

    # global acceleration 
    a_g = d2x + np.array([0, 0, g])
    thrust = m * np.linalg.norm(a_g) 

    # --- rotation ---
    
    # axis of body frame in global freference frame: [xb, yb, zb]^T

    zb = a_g / np.linalg.norm(a_g)  # z-axis of body frame in global body reference frame

    xc = np.array([np.cos(psi), np.sin(psi), 0])  # desire heading - yaw in global refernce frame
    
    yb_unit = np.cross(zb, xc) # y-axis of body frame in global bofy refernce frame - unit (NA ODWRÓT)
    yb = yb_unit/np.linalg.norm(yb_unit) # y-axis of body frame in global bofy refernce frame 

    xb = np.cross(yb, zb)

    rot_matrix = np.stack([xb, yb, zb], axis=1)

    q = R.from_matrix(rot_matrix).as_quat()
    
    q = np.array([q[3], q[0], q[1], q[2]]) # Scipy use [x, y, z, w] -> val data use [w, x, y, z]
    
    # --- angular velocity ---

    h = m/thrust * (d3x - np.dot(zb, d3x) * zb) # projection of jerk, projected on x-y plane (global ref frame)

    wx = -np.dot(h, yb)
    wy = np.dot(h, xb)
    wz = dpsi * np.dot(zw, zb)

    omega = np.array([wx, wy, wz])

    # --- torque ---
    mt_scale = m/thrust

    domega_x = -mt_scale * np.dot(d4x, yb) - 2 * mt_scale * np.dot(d3x, zb) * omega[0] + omega[1] * omega[2]
    domega_y = mt_scale * np.dot(d4x, xb) - 2 * mt_scale * np.dot(d3x, zb) * omega[1] - omega[0] * omega[2]
    
    domega_z = d2psi * np.dot(zw, zb) + dpsi * np.dot(zw, np.cross(omega, zb))

    domega = np.array([domega_x, domega_y, domega_z])

    torque = np.dot(J, domega) + np.cross(omega, np.dot(J, omega))

    # --- compose state vector ----

    state = np.concatenate([x, q, dx, omega])
    
    control_input = np.concatenate([np.array([thrust]), torque]) 

    return state, control_input



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