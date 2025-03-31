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

kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential


dt = 0.001

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
pT = p_train[:,-1] 

Q0 = Q_train[:,0] 
QT = Q_train[:,-1]


dmpTask = dmpSE3(N_in=W[0,:].size, T_in=T)
dmpTask.set_weights(W, T, Q0=Q0, Qtarget=QT)

dmpTask.set_goal(pT, QT)
dmpTask.set_init_pose(p0, Q0)


dmpTask.set_tau(1)

dmpTask.plotResponse(dt,p0,Q0,math.floor(T/dt))

plt.show()
# # plot results
# fig = plt.figure(figsize=(4, 7))

# fig.suptitle('Training dataset')

# for i in range(3):
#     axs = fig.add_axes([0.21, ((5-i)/6)*0.8+0.2, 0.7, 0.11])
#     axs.plot(t, p_train[i, :], 'k', linewidth=1.0)
#     axs.set_xlim([0, t[-1]])
#     axs.set_ylabel('$p_' + str(i+1) + '(t)$ [m]',fontsize=14 )
#     axs.set_xticks([])

# for i in range(4):
#     axs = fig.add_axes([0.21, ((5-(i+3))/6)*0.8+0.2, 0.7, 0.11])
#     axs.plot(t, Q_train[i, :], 'k', linewidth=1.0)
#     axs.set_xlim([0, t[-1]])
#     axs.set_ylabel('$Q_' + str(i+1) + '(t)$ [rad]',fontsize=14 )
    
#     if i==3:
#         axs.set_xlabel('Time (s)',fontsize=14 )
#     else:
#         axs.set_xticks([])

# plt.show()


