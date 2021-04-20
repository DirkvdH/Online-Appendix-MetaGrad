
import numpy as np
import scipy as sp
import cvxpy as cvx
import domains as dommi

#%%

class AdaGrad:
    def __init__(self, domain, eta = "auto", delta = 1e-8, outcome = "last"):
        self.domain = domain
        self.eta = eta
        if eta == "auto":
            self.eta = 1/3 * domain.radius # undo overscaling of domain by factor 3
        self.hessian = np.repeat(delta, domain.dimension)
        self.weights = domain.center(1).reshape(domain.dimension)
        self.LACeLw = domain.center(1).reshape(domain.dimension)
        self.domain.diag = True
        self.outcome = outcome
        # TODO: leg uit waarom we een nieuw domein nodig hebben ipv self.domain
        self.DiagDom = dommi.LinfBall(domain.radius, domain.dimension, diagonal = True)
        self.insideLACeL = True
        
        
    def getweights(self):
        if self.domain.name == "LACeL":
            return(self.LACeLw)
        return(self.weights)
    
    def update(self, gradient):
        gradi = gradient
        self.hessian = self.hessian + gradi**2
        wtilde = self.weights - self.eta / np.sqrt(self.hessian) * gradi
        if not self.domain.testdomain(wtilde):
            if self.domain.name == 'linfball':
                self.weights = self.DiagDom.project(wtilde, self.hessian).reshape(self.domain.dimension)
            else:
                self.weights = self.DiagDom.project(wtilde, np.diag(np.sqrt(self.hessian))).reshape(self.domain.dimension)
        else:
            self.weights = wtilde
    
    def getname(self):
        return("AdaGrad")

    def LACeLupdate(self, datarow):
        if self.outcome == "last":
            xt = datarow[0:-1]
        if self.outcome == "first":
            xt = datarow[1:]
        if self.outcome == "none":
            xt = datarow
        self.insideLACeL = self.domain.testdomainpred(self.weights, xt)
        if not self.insideLACeL:
            self.LACeLw = self.domain.futureproject(self.weights, np.diag(np.repeat(1, self.domain.dimension)), xt)
        else:
            self.LACeLw = self.weights
        
    def effectiveDim(self):
        return(np.sum(np.sqrt(1/self.hessian))/np.max(np.sqrt(1/self.hessian)))