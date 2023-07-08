import numpy as np

class interpolation_polynomial:
    def __init__(self, x_sample, y_sample):
        self.x_sample=x_sample
        self.y_sample=y_sample
        self.coefs=[]
        self.compute_coefs()
        
    @staticmethod
    def polynomial_terms(x, num_terms):
        return np.flip(np.array([np.power(x, i) for i in range(num_terms)]), 0).T
        
    def interpolate(self, x):
        terms = self.polynomial_terms(x, len(self.coefs))
        y = np.dot(terms,self.coefs)
        return y
    
    def compute_coefs(self):
        A = self.polynomial_terms(self.x_sample, len(self.x_sample))
        self.coefs = np.dot(np.linalg.inv(A), self.y_sample)
        return self.coefs