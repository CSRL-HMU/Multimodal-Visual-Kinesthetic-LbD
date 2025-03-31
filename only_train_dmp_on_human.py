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

folderPath = pathlib.Path(__file__).parent.resolve()

if os.name == 'nt': # the OS is Windows
    data = scipy.io.loadmat(str(folderPath) +'\\Logging_human.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/Logging_human.mat')


# x_train = data['x_data']

p_train = np.array(data['plog']).T
Q_train = np.array(data['Qlog']).T
Q_train = makeContinuous(Q_train)

dt = 0.002
t = np.array(list(range(p_train[1,:].size))) * dt

print("T=", t[-1])

Nsamples = p_train[1,:].size



R0 = quat2rot(Q_train[:,0])
p0 = np.copy(p_train[:,0])

for i in range(Nsamples):
  # print(i)
  p_train[:,i] = R0.T @ (p_train[:,i] - p0)
  Rch = quat2rot(Q_train[:,i])
  Q_train[:,i] = rot2quat(R0.T @ Rch)



dmpTask = dmpSE3(N_in=20, T_in=t[-1])


dmpTask.train(dt, p_train, Q_train, True)




print("Training completed. The dmp_model is saved.")

################# UNCOMMENT THIS FOR SEVING THE weights FILE 
# mdic = {"W": dmpTask.get_weights(), "T": dmpTask.T}
# scipy.io.savemat("dmp_model.mat", mdic)


plt.show()


Q_array = makeContinuous(Q_train)
p_array = p_train.copy()
mdic = {"p_array": p_array[:,1:], "Q_array": Q_array[:,1:], "dt": dt}
scipy.io.savemat("training_demo_human.mat", mdic)



kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential

folderPath = pathlib.Path(__file__).parent.resolve()

if os.name == 'nt': # the OS is Windows
    data = scipy.io.loadmat(str(folderPath) +'\\training_demo_human.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/training_demo_human.mat')

# x_train = data['x_data']

p_train = np.array(data['p_array'])
Q_train = np.array(data['Q_array'])

dt = data['dt'][0][0]
t = np.array(list(range(p_train[0,:].size))) * dt

dmpTask = dmpSE3(20, t[-1])

print("dmpTask.T=", dmpTask.T)
dmpTask.train(dt, p_train, Q_train, False)

mdic = {"W": dmpTask.get_weights(), "T": dmpTask.T}
scipy.io.savemat("dmp_model_human.mat", mdic)

print("Training completed. The dmp_model is saved.")
