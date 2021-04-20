
import numpy as np
import scipy as sp
import cvxpy as cvx


#%%
class L2Ball:
    def __init__(self, radius, dimension):
        self.name = 'l2ball'
        self.radius = radius
        self.dimension = dimension
        self.diameter = 2 * radius
      
    def testdomain(self, weights):
        return(np.sqrt(np.dot(weights, weights)) <= self.radius)
    
    def Gnorm(self, G):
        return(np.sqrt(np.dot(G, G)))
    
    def initialize(self, MGobj):
        None
        
    def project(self, weights, hessian):
        x2 = cvx.Variable(self.dimension)
        constraints = [cvx.norm(x2, 2) <= self.radius]
        obj = cvx.Minimize(cvx.quad_form(x2 - weights, hessian)) 
        prob = cvx.Problem(obj, constraints)
        try:
            prob.solve()
        except:
            prob.solve(solver = cvx.SCS)
        return(np.array(x2.value))
    
    def center(self, netas):
        return(np.zeros((netas, self.dimension)))

class LinfBall:
    def __init__(self, radius, dimension, diagonal = False):
        self.name = 'linfball'
        self.radius = radius
        self.dimension = dimension
        self.diameter = 2  * radius
        self.diagonal = diagonal
    
    def initialize(self, MGobj):
        None
    
    def testdomain(self, weights):
        return(np.max(np.abs(weights)) <= self.radius)
     
    def Gnorm(self, G):
        return(np.sum(np.abs(G)))
        
    def project(self, weights, hessian):
        if self.diagonal:
            return(np.minimum(np.abs(weights), self.radius) * np.sign(weights))
        else:
            x2 = cvx.Variable(self.dimension)
            constraints = [cvx.norm(x2, 'inf') <= self.radius]
            obj = cvx.Minimize(cvx.quad_form(x2 - weights, hessian)) 
            prob = cvx.Problem(obj, constraints)   
            try:
                prob.solve()
            except:
                prob.solve(solver = cvx.SCS) 
            return(np.array(x2.value))
    
    def center(self, netas):
        return(np.zeros((netas, self.dimension)))
    
class L1Square:
    def __init__(self, radius, dimension):
        self.name = 'l1square'
        self.radius = radius
        self.dimension = dimension
        self.diameter = 2 * radius
    
    def initialize(self, MGobj):
        None
    
    def testdomain(self, weights):
        return(np.sum(np.abs(weights)) <= self.radius)
    
    def Gnorm(self, G):
        return(np.max(np.abs(G)))
    
    def project(self, weights, hessian):
        x2 = cvx.Variable(self.dimension)
        constraints = [cvx.norm(x2, 1) <= self.radius]
        obj = cvx.Minimize(cvx.quad_form(x2 - weights, hessian)) 
        prob = cvx.Problem(obj, constraints)   
        try:
            prob.solve()
        except:
            prob.solve(solver = cvx.SCS) 
        return(np.array(x2.value))
    
    def center(self, netas):
        return(np.zeros((netas, self.dimension)))
    
    

    
    
class LACeL: #Luo, Agarwal, Cesa-Bianchi, and Langford (2016)
    def __init__(self, radius, dimension, C):
        self.name = 'LACeL'
        self.C = C
        self.radius = radius
        self.dimension = dimension
        self.diameter = 2*radius
      
    def testdomain(self, weights):
        return(True)
    
    def testdomainpred(self, weights, x):
        return(np.abs(np.dot(weights, x)) <= self.C)
    
    def Gnorm(self, G):
        return(np.sqrt(np.dot(G, G)))
    
    def initialize(self, MGobj):
        None
        
    def project(self, weights, hessian):
        return(weights)
    
    def futureproject(self, weights, inversehessian, xt):
        tau = np.sign(np.dot(weights, xt)) * np.max([np.abs(np.dot(weights, xt)) - self.C, 0])
        division = np.dot(np.matmul(xt, inversehessian), xt)
        if division == 0:
            return(weights)
        newweights = weights -  tau * 1/(division) * np.matmul(inversehessian, xt)
        return(newweights)
    
    def center(self, netas):
        return(np.zeros((netas, self.dimension)))