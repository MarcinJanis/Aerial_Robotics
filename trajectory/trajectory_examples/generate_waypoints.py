import numpy as np
import pandas as pd 
import math 



def generate_waypoints(file_name, eq_x, eq_y, eq_z, n):

    t = np.arange(n)
    x = eq_x(t)
    y = eq_y(t)
    z = eq_z(t)

    pts = np.column_stack((x, y, z))

    header = 'x,y,z'

    try:
        np.savetxt(file_name, pts, delimiter=',', header=header, comments='')
        return True
    except:
        return False



def execute():

    file_name = '/home/developer/ros2_ws/src/trajectory/trajectory_examples/circle.txt'

    pts_num = 20

    # define funcitons
    def eqx(n):
        return 2*np.sin(2*np.pi/pts_num*n)
    def eqy(n):
        return 3*np.cos(2*np.pi/pts_num*n)
    def eqz(n):
        return np.ones_like(n)

    result = generate_waypoints(file_name, eqx, eqy, eqz, pts_num)

    if result:
        print(f'Wwaypoints generated: {file_name}')
    else:
        print(f'Error while generating waypoints')


execute()   

