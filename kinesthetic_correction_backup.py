from kernel import *
from kernelBase import *
from CSRL_math import *
from CSRL_orientation import *
from dmpSE3 import *
from dmpR3 import *
from dmpSO3 import *
import scipy.io
import pathlib
import os
import time
from dmp import *

import roboticstoolbox as rt
import rtde_receive
import rtde_control 
import math 
import scienceplots
import getch

from spatialmath import SE3


kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential


# Gains
K = 4.0 * np.identity(6)
K[-3:, -3:] = 5.0 * K[-3:, -3:]  # Orientation gains

dt = 0.002

########### Test SE(3) dmp
folderPath = pathlib.Path(__file__).parent.resolve()


data = scipy.io.loadmat(str(folderPath) +'/dmp_model.mat')

# x_train = data['x_data']

W = np.array(data['W'])
T = data['T'][0][0]

# print(N)

train_data = scipy.io.loadmat(str(folderPath) +'/training_demo.mat')


p_train = np.array(train_data['p_array'])
Q_train = np.array(train_data['Q_array'])

p0 = p_train[:,0] 
pT = 0.4 * p_train[:,-1] 

Q0 = Q_train[:,0] 
QT = Q_train[:,-1]

dmpTask = dmpSE3(N_in=W[0,:].size, T_in=T)
dmpTask.set_weights(W, T, Q0=Q0, Qtarget=QT)

dmpTask.set_goal(pT, QT)
dmpTask.set_init_pose(p0, Q0)


dmpTask.set_tau(1)

#define UR3e
rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")


#declare math pi 
pi = math.pi 

#define the robot with its DH parameters 
ur = rt.DHRobot([
    rt.RevoluteDH(d = 0.1807, alpha = pi/2),
    rt.RevoluteDH(a = -0.6127),
    rt.RevoluteDH(a = -0.57155),
    rt.RevoluteDH(d = 0.17415, alpha = pi/2),
    rt.RevoluteDH(d = 0.11985, alpha = -pi/2),
    rt.RevoluteDH(d = 0.11655)
], name='UR10e')

# # Define the tool
tool_position = SE3(0.0, 0.0, 0.12)  # Position of the tool (in meters)


ur.tool = tool_position 

################ Move robot to initial pose

rtde_c.teachMode()  

print('Press any key...')
char = getch.getch() 

rtde_c.endTeachMode()

################ Read robot's initial pose

q0 = np.array(rtde_r.getActualQ())

#get initial end effector position 
g0rob = ur.fkine(q0)
R0rob = np.array(g0rob.R)
p0rob = np.array(g0rob.t)
Q0rob = rot2quat(R0rob)

#initialize DMP state

# print('p0=', p0)
pd = p0.copy()  # Initially, setting the desired position to the initial position p0
dot_pd = np.zeros(3)

ddp = np.zeros(3)
dp = np.zeros(3)
ddeo = np.zeros(3)
deo = np.zeros(3)

Q_desired = Q0.copy()  # Initially, setting the desired orientation to the initial orientation R0
Q = Q0.copy()

eo = logError(QT, Q0)  # Initial orientation error
dot_eo = np.zeros(3)
ddot_eo = np.zeros(3)
omegad = np.zeros(3)
dot_omegad = np.zeros(3)

z = 0.0   #phase variable
dz = 0.0

t = 0

#start logging 
plog = np.copy(p0rob)
pdlog = np.copy(p0rob)
tlog = t
Qlog = np.copy(Q0rob)
Qd_log = np.copy(Q0rob)


i = 0
while t < T:
    i = i + 1

    t_now = time.time()
    t_start = rtde_c.initPeriod()


    # Integrate time
    t = t + dt

    # Euler integration to update the states
    z += dz * dt
    pd += dp * dt   
    eo += deo * dt
    dot_eo += ddeo * dt
    dot_pd += ddp * dt

    

     
   
    # Calculate DMP state derivatives (get z_dot, p_dot, pdot_dot)
    # dz, dp, ddp = model.get_state_dot(z, pd, dot_pd)
    # get state dot
    dz, dp, ddp, deo, ddeo = dmpTask.get_state_dot(   z, 
                                                    pd,                                                                                      
                                                    dot_pd, 
                                                    eo,
                                                    dot_eo)
    
    
    Q_desired =  quatProduct( quatInv( quatExp( 0.5 * eo ) ) , QT )
    omegad = logDerivative2_AngleOmega(dot_eo, quatProduct(Q_desired,quatInv(QT)))
    omegad = quat2rot(QT) @ omegad


    # translate everything to the world frame
    pdw = p0rob + R0rob @ pd
    Qdw = rot2quat(R0rob @ quat2rot(Q_desired))
    dot_pdw = R0rob @ dot_pd
    omegadw = R0rob @ omegad

    # Get the actual joint values 
    q = np.array(rtde_r.getActualQ())

    # Get  end-efector pose
    g = ur.fkine(q)
    R = np.array(g.R)
    p = np.array(g.t)

    
    # get full jacobian
    J = np.array(ur.jacob0(q))

    # get translational jacobian
    # Jp = J[:3, :] 

    # pseudoInverse
    Jinv = np.linalg.pinv(J)

    # velocity array
    velocity_matrix = np.hstack((dot_pdw, omegadw))
    
    # error matrix 
    Q = rot2quatCont(R,Q)
    eo_robot = logError(Q, Qdw)
    error_matrix = np.hstack((p - pdw, eo_robot))        


    # tracking control signal
    qdot = Jinv @ (velocity_matrix - K @ error_matrix)
    # qdot = np.zeros(6)
    
    # set joint speed
    rtde_c.speedJ(qdot, 10.0, dt)

    # log data
    tlog = np.vstack((tlog, t))
    plog = np.vstack((plog, p))
    pdlog = np.vstack((pdlog, pdw))
    
    Qlog = np.vstack((Qlog, rot2quat(R)))
    Qd_log = np.vstack((Qd_log, Qdw))

    # synchronize
    rtde_c.waitPeriod(t_start)
    
# close control
rtde_c.speedStop()
rtde_c.stopScript()


#plot the training data
fig = plt.figure(figsize=(10, 8))
for i in range(3):
    axs = fig.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
    axs.plot(tlog, pdlog[:,i], 'r--', label='DMP p')
    axs.plot(tlog, plog[:,i], 'k', label='Executed Motion p')
    axs.set_ylabel('p'+str(i+1))
    axs.set_xlabel('Time (s)')
    axs.legend()
    axs.grid(True)
plt.show()

# plot the quaternion training data
fig = plt.figure(figsize=(10, 8))
for i in range(4):
    axs = fig.add_subplot(4, 1, i+1)
    axs.plot(tlog, Qd_log[:,i], 'r--', label='DMP Q')
    axs.plot(tlog, Qlog[:,i], 'k', label='Executed Motion Q')
    axs.set_ylabel('Q'+str(i+1))
    axs.set_xlabel('Time (s)')
    axs.legend()
    axs.grid(True)
plt.show()


