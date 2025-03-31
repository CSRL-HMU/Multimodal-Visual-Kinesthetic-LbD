import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
import rtde_receive
import rtde_control
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import scipy.io as scio
import time
import keyboard
from CSRL_math import *
from CSRL_orientation import *
import cv2


# open HMI
img = np.zeros((100,100))
cv2.imshow('HMI',img)

print('Press a to continue')
# blocking
while True:
    check = cv2.waitKey(1)
    if check == ord('a'):
        break

# Define leader (UR3e)
rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

# Get initial configuration
q0 = np.array(rtde_r.getActualQ())

# Declare math pi
pi = math.pi

# Create the robot 

# Define the kinematics of the leader robot (UR3e)
robot_l = rt.DHRobot([
    rt.RevoluteDH(d = 0.15185, alpha = pi/2),
    rt.RevoluteDH(a = -0.24355),
    rt.RevoluteDH(a = -0.2132),
    rt.RevoluteDH(d = 0.13105, alpha = pi/2),
    rt.RevoluteDH(d = 0.08535, alpha = -pi/2),
    rt.RevoluteDH(d = 0.0921)
], name='UR3e')

# Control cycle of the leader
dt = 0.002

# Init time
t = 0.0

# Start logging
qlog = q0
tlog = t

# get time now
t_now = time.time()

# Tool mass
tool_mass = 0.145

# gravity acceleration
gAcc = 9.81


freq = 0.5

# initialize task space variables (admittance simulation)
pddot = np.zeros(3)
pdot = np.zeros(3)
v_total = np.zeros(6)
vdot_total = np.zeros(6)
wddot = np.zeros(3)
wdot = np.zeros(3)

# Define the admittance inertia
M = np.identity(6)
l = 0.1
M[0:3, 0:3] = 4.0 * M[0:3, 0:3]
M[3:6, 3:6] = l * l * M[0, 0] * M[3:6, 3:6]

# Compute inverse (constant)
Minv = np.linalg.inv(M)

# Define the admittance damping
D = np.identity(6)
D[0:3, 0:3] = 20.0 * D[0:3, 0:3]
D[3:6, 3:6] =  0.2 * D[3:6, 3:6]

# Define the admittance stiffness
K = np.identity(6)
K[0:3, 0:3] = 100.0 * K[0:3, 0:3]
K[3:6, 3:6] =  1.0 * K[3:6, 3:6]

# initialize leader Jacobian
J = np.zeros([6, 6])

# get force measurement bias
f_offset_tool = np.array(rtde_r.getActualTCPForce()) #- tool_mass * gAcc

# Get pose of the leader's wrist  
g = robot_l.fkine(q0)
R = np.array(g.R)
p = np.array(g.t)

# Desired pose
pd = p
quatd = rot2quat(R) 

print('---------Robot pose:')
print('R: ', R)
print('p: ', p)

# Initialize logging data -----------
pl_log = p
Ql_log = rot2quat(R) # Quaternion format
def_log = np.zeros(2) # Deformation`
Fh_log = np.zeros(6) # Force applied by human


# Main control loop ..... runs 50000 times (100s)
for i in range(50000):

    # this is to synchronise with the UR3e 
    t_start = rtde_c.initPeriod()


    # Get joint values
    q = np.array(rtde_r.getActualQ())


    # Calculate the pose of the leader's wrist 
    g = robot_l.fkine(q)
    R = np.array(g.R)
    p = np.array(g.t)
    quat = rot2quat(R)

    
    # Integrate time
    t = t + dt

    # End-effector Jacobian of the leader 
    J = np.array(robot_l.jacob0(q))

    # Get the force/torque measurement from the sensor
    f = np.array(rtde_r.getActualTCPForce()) - f_offset_tool

    # The force/torque with respect to the wrist of the leader robot 
    f[:3] = f[:3]
    f[-3:] = f[-3:]
    
    # Dead-zone (thresholding) of the the measurement of the force
    fnorm = np.linalg.norm(f[:3])
    if fnorm > 0.001:
        nF = f[:3] / fnorm
        if fnorm<4.0:
            f[:3] = np.zeros(3)
        else:
            f[:3] = f[:3] - 4.0 * nF

        # Dead-zone (thresholding) of the the measurement of the torqeu
        taunorm = np.linalg.norm(f[-3:])
        nTau = f[-3:] / taunorm
        if taunorm < 0.5:
            f[-3:] = np.zeros(3)
        else:
            f[-3:] = f[-3:] - 0.5 * nTau



    #  Integrate the task space veloities of the leader (admittance control)
    v_total = v_total + vdot_total * dt

    # State variables for orientation
    e = logError(quat, quatd)
    
    # Circle path in the task space centered on the starting point of the robot
    pr = [0.2 * np.cos(2 * np.pi * t * freq) + pd[0] - 0.2,
          pd[1],
          0.1 * np.sin(2 * np.pi * t * freq) + pd[2]]

    pr_dot = [- 2 * np.pi * freq * 0.2 * np.sin(2 * np.pi * t * freq),
          0,
          2 * np.pi * freq * 0.1 * np.cos(2 * np.pi * t * freq)]

    pr_2dot = [- 2 * np.pi * freq * 2 * np.pi * freq * 0.2 * np.cos(2 * np.pi * t * freq),
              0,
              - 2 * np.pi  * freq * 2 * np.pi * freq * 0.1 * np.sin(2 * np.pi * t * freq)]



    e_total = np.concatenate((p - pr , e))

    vr_total = np.concatenate(( pr_dot, np.zeros(3)))
    vrdot_total = np.concatenate(( pr_2dot, np.zeros(3)))

    vdot_total = vrdot_total + Minv @ (- D @ ( v_total - vr_total) - K @ e_total + f)

    # Set the joint velocity of the leader
    qdot_command = np.linalg.pinv(J, 0.1) @ v_total
    # print('qdot_command', qdot_command)
    # qdot_command = np.zeros(6)
    rtde_c.speedJ(qdot_command, 3.0, dt)

    # Log variables
    pl_log = np.vstack((pl_log, p))
    Ql_log = np.vstack((Ql_log, rot2quat(R)))

    Fh_log = np.vstack((Fh_log, f))
    tlog = np.vstack((tlog,t))

    check = cv2.waitKey(1)
    if check & 0xFF == 27:
        break

    # wait for the period synch of the UR3e
    rtde_c.waitPeriod(t_start)



# write the data to a file 
data = {'pl_log': pl_log, 'Ql_log': Ql_log, 'Fh_log': Fh_log, 'tlog': tlog}
scio.savemat('Logging.mat', data)

# Stop robot 
rtde_c.speedStop()
rtde_c.stopScript()
