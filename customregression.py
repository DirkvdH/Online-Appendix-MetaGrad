

# based on https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
#%%

def abslossF(beta, X, y):
    return(np.sum(np.abs(np.matmul(X, beta).flatten() - y)))
    
def abslossFgrad(beta, X, y):
    sgn = np.sign(np.matmul(X, beta).flatten() - y.flatten())
    sgn.shape = (X.shape[0], 1)
    grad = X*sgn
    return(np.sum(grad, axis = 0))
    
    
def hingelossF(beta, X, y):
    yhat = np.matmul(X, beta)
    temphinge = 1 - yhat.flatten() * y.flatten()
    temphinge = temphinge.flatten()
    hingeopt = np.maximum(0, temphinge)
    return(np.sum(hingeopt))    
    
    
def hingelossFgrad(beta, X, y):
    yhat = np.matmul(X, beta)
    yproduct = yhat.flatten() * y.flatten()
    temphinge = 1 - yproduct
    temphinge = temphinge.flatten()
    hingeopt = np.maximum(0, temphinge)
    sel = hingeopt != 0
    y.shape = (X.shape[0], 1)
    gradtemp = -X * y 
    y = y.flatten()
    return(np.sum(gradtemp[sel, ], axis = 0))


class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, loss_function = abslossF, grad_function = abslossFgrad,
                 X = None, Y = None, sample_weights = None, beta_init = None,
                 optimizer = 'BFGS', regularization = 0, optimizer2 = 'Nelder-Mead'):
        self.beta = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.beta_init = beta_init
        self.grad_function = grad_function
        self.X = X
        self.Y = Y
        self.optimizer = optimizer
        self.optimizer2 = optimizer2
        self.regularization = regularization
            
    
    def predict(self, X):
        prediction = np.matmul(X, self.beta)
        return(prediction)

    def model_error(self):
        error = self.loss_function(self.beta, self.X, self.Y)
        return(error)
    
    def regloss(self, beta, X, y):
        return(self.loss_function(beta, X, y) + self.regularization * np.sum(beta**2))
    
    def gradregloss(self, beta, X, y):
        return(self.grad_function(beta, X, y) + self.regularization * 2 * beta)
    
    def fit(self, maxiter = 250):        
        if type(self.beta_init) == type(None):
            self.beta_init = np.repeat(0, self.X.shape[1])
        else: 
            pass
            
        if self.beta != None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")
            
        res = minimize(fun = self.regloss, x0 = self.beta_init,
                       method = self.optimizer, 
                       options={'maxiter': maxiter, 'disp': True},
                       jac = self.gradregloss, args = (self.X, self.Y))
        oldbeta = res.x
        self.beta = oldbeta
        self.beta_init = self.beta
        self.converge = res.nit <= maxiter
        