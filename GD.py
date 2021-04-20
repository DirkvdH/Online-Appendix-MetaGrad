

import numpy as np
import scipy as sp
import cvxpy as cvx

#%%

class GradientDescentnorm:
    def __init__(self, domain, eta, delta = 1e-8, outcome = "last"):
        self.domain = domain
        self.eta = eta
        if eta == "auto":
            sys.exit("Automatic setting of eta not available")
        self.hessian = np.repeat(delta, domain.dimension)
        self.weights = domain.center(1).reshape(domain.dimension)
        self.domain.diag = True
        self.outcome = outcome
        self.name = "GDnorm"
    
    def getweights(self):
        return(self.weights)
            
    def update(self, gradient):
        self.hessian = self.hessian + np.dot(gradient, gradient)
        wtilde = self.weights - self.eta / np.sqrt(self.hessian) * gradient
        if not self.domain.testdomain(wtilde):
            if self.domain.name == 'linfball':
                self.weights = self.domain.project(wtilde, self.hessian).reshape(self.domain.dimension)
            else:
                self.weights = self.domain.project(wtilde, np.diag(np.sqrt(self.hessian))).reshape(self.domain.dimension)
        else:
            self.weights = wtilde
    
    def getname(self):
        return(self.name)
    
    def LACeLupdate(self, datarow):
        if self.outcome == "last":
            xt = datarow[0:-1]
        if self.outcome == "first":
            xt = datarow[1:]
        if self.outcome == "none":
            xt = datarow
        
        self.weights = self.domain.futureproject(self.weights, np.diag(1/np.sqrt(self.hessian)), xt)  
        
    def effectiveDim(self):
        return(1/self.domain.dimension)

class GradientDescentt:
    def __init__(self, domain, G, eta = "auto", delta = 1e-8, Ghat = False, Gupdate = False, outcome = "last"):
        self.domain = domain
        self.eta = eta
        if eta == "auto":
            sys.exit("Automatic setting of eta not available")
        self.hessian = np.repeat(delta, domain.dimension)
        self.weights = domain.center(1).reshape(domain.dimension)
        self.domain.diag = True
        self.G = G
        self.Ghat = Ghat
        self.Gupdate = Gupdate
        self.name = "GDt"
        self.outcome = outcome
        self.delta = delta
    
    def getweights(self):
        return(self.weights)
            
    def update(self, gradient):
        if self.Ghat:
            self.Ghat = False
            self.G = np.max([np.sqrt(np.dot(gradient, gradient)), self.delta])
        if self.Gupdate:
            tempG = np.sqrt(np.dot(gradient, gradient))
            if self.G < tempG:
                self.G = tempG
        self.hessian = self.hessian + 1
        wtilde = self.weights - self.eta / (self.G * np.sqrt((self.hessian))) * gradient
        if not self.domain.testdomain(wtilde):
            if self.domain.name == 'linfball':
                self.weights = self.domain.project(wtilde, self.hessian).reshape(self.domain.dimension)
            else:
                self.weights = self.domain.project(wtilde, np.diag(self.G*np.sqrt(self.hessian))).reshape(self.domain.dimension)
        else:
            self.weights = wtilde
    
    def getname(self):
        return(self.name)

    def LACeLupdate(self, datarow):
        if self.outcome == "last":
            xt = datarow[0:-1]
        if self.outcome == "first":
            xt = datarow[1:]
        if self.outcome == "none":
            xt = datarow
        
        self.weights = self.domain.futureproject(self.weights, np.diag(1 / (self.G * np.sqrt((self.hessian)))), xt)
    
    def effectiveDim(self):
        return(1/self.domain.dimension)
    

  