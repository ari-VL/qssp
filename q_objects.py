import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a-b)<(atol+rtol*np.abs(b))

def expect(mOp, state):
    #expectation value of measuring state with mop
    return np.real(np.trace(state @ mOp))

class qstate:
    """A discrete state in a Hilbert space
    
    Attributes
    ----------
    dim : integer
        dimension of the Hilbert space
    state: complex numpy array
        density matrix describing the quantum state

    Methods
    -------
    is_normalized()
        Returns True if state is normalized array, False otherwise
    normalize()
        Normalizes state if not already normalized
    is_hermitian()
        Returns True if state is Hermitian array
    is_positive()
        Returns True if all eigenvalues of state matrix are 0 or positive, False otherwise
    is_valid()
        Returns True if state is normalized, Hermitian and postive, False otherwise
    is_pure()
        Returns True if state matrix is approximately idempotent
    vn_entropy(base=2)
        Returns von Neumann entropy of state matrix with desired base
    measure(measurement, labels=False)
        Applies measurement to state and returns probabilities of different measurement outcomes
    measure_sample(measurement, num_samples)
        Returns measurement outcomes when applying measurement sampled 'num_samples' times with correct probabilities
    """

    def __init__(self, amplitudes):
        """
        Parameters
        ----------
        amplitudes: numpy array
            Array describing quantum state as vector (pure) or matrix (mixed)
        """

        amplitudes = amplitudes.astype(complex)
        
        #if amplitudes is ket --> dm
        if amplitudes.ndim == 1:
            self.dim = len(amplitudes)
            self.state = np.outer(amplitudes.conj().T,amplitudes)
            if(not self.is_normalized()):
                print("ERROR: Not normalized")
        
        #if amplitudes is matrix -> dm
        elif amplitudes.ndim >= 2:
            self.dim = len(amplitudes[0])
            self.state = amplitudes
            if(not self.is_normalized()):
                print("ERROR: Not normalized")
        
        else:
            print("ERROR: Invalid input dimensions")
        
    def is_normalized(self):
        """ Returns True if state is normalized array, False otherwise."""

        if near(np.trace(self.state),1):
            return True
        else:
            return False
        
    def normalize(self):
        """Normalizes state if not already normalized."""

        if self.is_normalized():
            pass
        else:
            norm = np.trace(self.state)
            self.state = self.state/norm
            pass
    
    def is_hermitian(self):
        """Returns True if state is Hermitian array."""

        return np.allclose(self.state, np.conj(self.state.T))
    
    def is_positive(self):
        """Returns True if all eigenvalues of state matrix are 0 or positive, False otherwise."""
       
        eigs = la.eig(self.state)[0]
        
        return not any(not near(eig,0) and eig < 0 for eig in eigs)
        
    def is_valid(self):
        """Returns True if state is normalized, Hermitian and postive, False otherwise."""
        
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
        """Returns True if state matrix is approximately idempotent."""
        
        if self.is_normalized():
            return (near(np.trace(np.dot(self.state,self.state)),1))
        else:
            print('Not Normalized')
            return False
        
    def vn_entropy(self,base=2):
        """Returns von Neumann entropy of state matrix with desired base.
        
        Parameters
        ----------
        base : real
            Base of logarithm to use (default is 2)
        """
        
        eigs = la.eig(self.state)[0]
        eigs = [0 if near(e,0) else e.real for e in eigs]
        return entropy(eigs,base=base)
    
    def measure(self, measurement, labels=False):
        """Applies measurement to state and returns probabilities of different measurement outcomes.
        
        Parameters
        ----------
        measurement : measurement class
            Measurement operators to apply to quantum state
        labels : Boolean
            If True, returns list of strings labeling measurement outcomes, default is False
        """
        
        probs = np.array([expect(mop, self.state) for mop in measurement.mOps])
        return probs
    
    def measure_sample(self, measurement, num_samples):
        """Returns measurement outcomes sampled with the correct probabilities.

        Parameters
        ----------
        measurement: instance of measurement class
        """

        outcomes = np.zeros(num_samples)
        for i in range(num_samples):
            outcome = np.random.choice(measurement.labels, p = self.measure(measurement))
            outcomes[i]=outcome
        return outcomes
    
class measurement:
    """A quantum measurement, consisting of a set of operators
    
    Attributes
    ----------
    mOps : list of numpy arrays 
        measurement operators in matrix form
    n_ops: int
        number of measurement operators
    tol_positvity: float
        numerical tolerance for checking positivity/completeness of measurement

    Methods
    -------
    is_postive()
        Returns True if all eigenvalues of measurement operators are non-negative.
    is_complete()
        Returns True if measurement operators sum to the identity (within tolerance).
    """

    def __init__(self, mOps, labels=None,tol_positivity=1e-8):
        """
        Parameters
        ----------
        mOps: list of numpy arrays
            Measurement operators in matrix form
        labels: list of strings
            List of labels corresponding to measurement outcomes, default is None
        tol_postivity: float
            Tolerance for checking completeness/positivity of measurement, default is 1e-8
        """

        #TODO: check sum to identity, check positivity
        self.mOps = mOps
        self.n_ops = len(self.mOps)
        if labels==None:
            self.labels = range(self.n_ops)
        else:
            self.labels=labels
        self.tol_positivity= tol_positivity

    def is_positive(self):
        """Returns True if all eigenvalues of measurement operators are non-negative, otherwise returns False."""

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
        """Returns True if measurement operators sum to the identity (within tolerance), otherwise returns False."""
        
        return np.allclose(np.sum(self.mOps,axis=0),np.identity(self.n_ops))
    
    #TODO: is_valid