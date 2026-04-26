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



