import numpy as np 
from util import interpolation_polynomial as poly

class regression:
    def __init__(self, x_sample, y_sample, lambda_=0, num_terms=3):
        self.x_sample=x_sample
        self.y_sample=y_sample
        self.coefs=[]
        
        # Hyperparameters
        self.lambda_=lambda_
        self.num_terms=num_terms
        
        self.compute_coefs()
        
    def compute_coefs(self):
        A = poly.interpolation_polynomial.polynomial_terms(self.x_sample, self.num_terms)
        # Gram matrix
        G = np.dot(A.T, A) + self.lambda_*np.identity(self.num_terms)
        self.coefs = np.dot(np.dot(np.linalg.inv(G), A.T), self.y_sample)
        return self.coefs
    
    def interpolate(self, x):
        terms = poly.interpolation_polynomial.polynomial_terms(x, len(self.coefs))
        y = np.dot(terms,self.coefs)
        return y