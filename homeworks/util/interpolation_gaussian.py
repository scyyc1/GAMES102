import numpy as np

class interpolation_gaussian:
    def __init__(self, x_sample, y_sample, b0=0, sigma=1):
        self.x_sample=x_sample
        self.y_sample=y_sample
        self.coefs=[b0]
        
        # Hyperparameters
        self.b0=b0
        self.sigma=sigma
        
        self.compute_coefs()
    
    @staticmethod
    def gaussian_terms(x, x_sample, num_terms, sigma=1):
        return np.array([interpolation_gaussian.gaussian(x, 1, x_sample[i], sigma) for i in range(num_terms)]).T
    
    @staticmethod
    def gaussian(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2 * c**2))
    
    def interpolate(self, x):
        terms = self.gaussian_terms(x, self.x_sample, len(self.coefs) - 1)
        y = np.dot(terms,self.coefs[1:]) + self.coefs[0]
        return y

    def compute_coefs(self):
        A = self.gaussian_terms(self.x_sample, self.x_sample, len(self.x_sample))
        self.coefs = np.concatenate((self.coefs, np.dot(np.linalg.inv(A), self.y_sample-self.b0)), axis=0)
        return self.coefs