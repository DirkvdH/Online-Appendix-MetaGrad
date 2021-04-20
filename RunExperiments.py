

import sys 
import RunLibsvm as RL
import numpy as np

#%%
if len(sys.argv) < 4:
    print("You need to provide all the arguments, arg1 = dataset, arg2 = lossfun, arg3 = algorithm/coefonly")
else:
    dataset = str(sys.argv[1])
    lossfun = str(sys.argv[2])
    alg = str(sys.argv[3])
    coefonly = alg == "coefonly" # if alg == coefonly, then it only computes the coefficient

    path = dataset + "+" + lossfun + "+" + alg + ".dat"
    RL.RunLibsvmPap(dataset, lossfun, [alg], path, coefonly)
