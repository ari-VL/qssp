import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a-b)<(atol+rtol*np.abs(b))

def expect(mOp, state):
    #expectation value of measuring state with mop
    return np.real(np.trace(state @ mOp))

class qstate:

    def __init__(self, amplitudes):
       
        amplitudes = amplitudes.astype(complex)
        
        #if amplitudes is ket --> dm
        if amplitudes.ndim == 1:
            self.dim = len(amplitudes)
            self.state = np.outer(amplitudes,amplitudes)
        
        #if amplitudes is matrix -> dm
        elif amplitudes.ndim >= 2:
            self.dim = len(amplitudes[0])
            self.state = amplitudes
        
        else:
            print("ERROR: Invalid input dimensions")
        
    def is_normalized(self):
        if near(np.trace(self.state),1):
            return True
        else:
            return False
        
    def normalize(self):
        if self.is_normalized():
            pass
        else:
            norm = np.trace(self.state)
            self.state = self.state/norm
            pass
    
    def is_hermitian(self):
        return np.allclose(self.state, np.conj(self.state.T))
    
    def is_positive(self):
        eigs = la.eig(self.state)[0]
        
        return not any(not near(eig,0) and eig < 0 for eig in eigs)
        
    def is_valid(self):
        normalized = self.is_normalized()
        hermitian = self.is_hermitian()
        positive = self.is_positive()
        
        if (normalized & hermitian & positive):
            return True
        else:
            if not normalized:
                print('Not Normalized')
            if not hermitian:
                print('Not Hermitian')
            if not positive:
                print('Not Positive')
            return False
        
    def is_pure(self):
        if self.is_normalized():
            return (near(np.trace(np.dot(self.state,self.state)),1))
        else:
            print('Not Normalized')
            return False
        
    def vn_entropy(self,base=2):
        eigs = la.eig(self.state)[0]
        eigs = [0 if near(e,0) else e.real for e in eigs]
        return entropy(eigs,base=base)
    
    def measure(self, measurement, labels=False):
        #A
        #returns a probability distribution, can return also a list of labels if labels=True
        probs = np.array([expect(mop, self.state) for mop in measurement.mOps])
        return probs
    
    def measure_sample(self, measurement, L):
        #A
        outcomes = np.zeros(L)
        for i in range(L):
            outcome = np.random.choice(measurement.labels, p = self.measure(measurement))
            outcomes[i]=outcome
        return outcomes
    
class measurement:
    
    def __init__(self, mOps, labels=None,tol_positivity=1e-8):
        #TODO: check sum to identity, check positivity
        self.mOps = mOps
        self.n_ops = len(self.mOps)
        if labels==None:
            self.labels = range(self.n_ops)
        else:
            self.labels=labels
        self.tol_positivity= tol_positivity

    
    # def is_projective(self):
    #     #check conditions or
    #     comp = self.is_complete()
    #     dim_check = (self.n_ops == len(self.mOps[0]))
    #     return (comp and dim_check)

    def check_positive(self):
        for mop in self.mOps:
            vs = la.eigvals(mop)
            for v in vs:
                #assert val.imag<tol_positivity
                #assert val.real >= -1*tol_positivity
                a = v.imag<self.tol_positivity
                b = v.real >= -1*self.tol_positivity
                if (a&b):
                    pass
                else:
                    return False
        return True

    def is_complete(self):
        return np.allclose(np.sum(self.mOps,axis=1),np.identity(self.n_ops))
    
    #TODO: is_valid