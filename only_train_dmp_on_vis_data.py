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
    data = scipy.io.loadmat(str(folderPath) +'\\training_demo.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/training_demo.mat')

# x_train = data['x_data']

p_train = np.array(data['p_array'])
Q_train = np.array(data['Q_array'])
Q_train = makeContinuous(Q_train)

dt = data['dt'][0][0]
t = np.array(list(range(p_train[1,:].size))) * dt

print("T=", t[-1])

dmpTask = dmpSE3(N_in=20, T_in=t[-1])


dmpTask.train(dt, p_train, Q_train, True)



print("Training completed. The dmp_model is saved.")

################# UNCOMMENT THIS FOR SEVING THE weights FILE 
# mdic = {"W": dmpTask.get_weights(), "T": dmpTask.T}
# scipy.io.savemat("dmp_model.mat", mdic)


plt.show()

