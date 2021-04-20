# Randomised check for the minimser that we found.


from RunLibsvm import load_data
import numpy as np
from math import *

dataname = "abalone_scale"
lossfun = "absolute"



def test_opt(dataname, lossfun):
    print("Processing " + dataname + " with " + lossfun)
    # get data
    data, lossf, gradf, domLinfalgs, domLACeL, domL2, d, G, comparator = load_data(dataname, lossfun, False)
    
    # get comparator coefficients
    what = np.load(dataname + "+" + lossfun + "+coef.npy")
    
    # compute loss and gradient
    loss = sum(lossf(what, row) for row in data)
    assert abs(sum(comparator) - loss) < 1e-7 # sanity check
    
    for scale in range(-6, 2):
        for rep in range(50):
            # get random close-by point (TODO worry this may be outside domain)
            woth = what + np.random.normal(scale=2**scale, size=np.shape(what))
            loth = sum(lossf(woth, row) for row in data)
            print(loth)
            # other point should be worse if 'what' is the minimiser
            assert loss <= loth, "Oops " + str(loth) + " should be below " + str(loss)
            
            
            
            
cdata = ["a9a",
         "australian_scale",
         "breast-cancer_scale",
         "covtype_scale",
         "diabetes_scale",
         "heart_scale",
         "ijcnn1",
         "ionosphere_scale",
         "phishing",
         "splice_scale",
         "w1atest",
         "w8a"
         ]
closses = [ 'hinge', 'logistic'] 

rdata = ["abalone_scale",
         "bodyfat_scale",
         "cpusmall_scale",
         "housing_scale",
         "mg_scale",
         "space_ga_scale"]
rlosses = [ 'squared', 'absolute'] 

for d in cdata:
    for l in closses:
        test_opt(d, l, False)
for d in rdata:
    for l in rlosses:
        test_opt(d, l, True)