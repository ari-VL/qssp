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
    dim : int
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

    def __init__(self, amplitudes, test_norm = True):
        """
        Parameters
        ----------
        amplitudes: numpy array
            Array describing quantum state as vector (pure) or matrix (mixed)
        """

        amplitudes = amplitudes.astype(complex)
        
        #if amplitudes is ket --> dm
        if amplitudes.ndim == 1:
            self.dim = int(len(amplitudes))
            self.state = np.outer(amplitudes.conj().T,amplitudes)
            if(test_norm and (not self.is_normalized())):
                print("WARNING: Not normalized")
        
        #if amplitudes is matrix -> dm
        elif amplitudes.ndim >= 2:
            self.dim = int(len(amplitudes[0]))
            self.state = amplitudes
            if(test_norm and (not self.is_normalized())):
                print("WARNING: Not normalized")
        
        else:
            raise ValueError("Invalid input dimensions")
    
    def __add__(self, state_2):
        """ Overloaded addition for qstates."""
        sum = self.state + state_2.state

        return qstate(np.array(sum))

    def __sub__(self, state_2):
        """ Overloaded subtraction for qstates."""
        diff = self.state - state_2.state

        return qstate(np.array(diff))

    def __mul__(self, num):
        """ Overloaded multiplication for qstates."""
        mult = num * self.state

        return qstate(np.array(mult), test_norm=False)

    def __rmul__(self, num):
        """ Overloaded reverse multiplication for qstates."""
        mult = self.state * num

        return qstate(np.array(mult), test_norm=False)

    def __truediv__(self, num):
        """ Overloaded division for qstates."""
        div = self.state / num

        return qstate(np.array(div), test_norm=False)

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
        sq_trace = np.trace(np.matmul(self.state,self.state))
        is_p = near(sq_trace,1)
        return is_p
##NOTE: DELETED THE FOLLOWING CODE, IT WAS INCORRECT. DO WE WANT A PRINT OPTION?
       # if self.is_normalized():
        #    return (near(np.trace(np.dot(self.state,self.state)),1))
       # else:
       #     print('Not Pure')
       #     return False
        
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

    def add_noise(self,noise_type,noise_level):
        """Returns state after passing it through a noisy channel.

        Parameters
        ----------
        noise_type: str
            Type of noisy channel to use, can be 'phaseflip', 'bitflip',
            'bitphaseflip', 'depolarizing', or 'amplitude_damping'
        noise_level: float
            Amount of noise to apply
        """

        if noise_type == 'phaseflip':
            E_0 = np.sqrt(1-noise_level)*np.array(([1,0],[0,1]))
            E_1 = np.sqrt(noise_level)*np.array(([1,0],[0,-1]))
            noisy_state = E_0 @ self.state @ (E_0.conj().T) + E_1 @ self.state @ (E_1.conj().T) 
        elif noise_type == 'bitflip':
            E_0 = np.sqrt(1-noise_level)*np.array(([1,0],[0,1]))
            E_1 = np.sqrt(noise_level)*np.array(([0,1],[1,0]))
            noisy_state = E_0 @ self.state @ (E_0.conj().T) + E_1 @ self.state @ (E_1.conj().T) 
        elif noise_type == 'bitphaseflip':
            E_0 = np.sqrt(1-noise_level)*np.array(([1,0],[0,1]))
            E_1 = np.sqrt(noise_level)*np.array(([0,-1j],[1j,0]))
            noisy_state = E_0 @ self.state @ (E_0.conj().T) + E_1 @ self.state @ (E_1.conj().T)
        elif noise_type == 'depolarizing':
            noisy_state = noise_level/2 * np.array([[1,0],[0,1]]) + (1-noise_level) * self.state
        elif noise_type == 'amplitude_damping':
            E_0 = np.array(([1,0],[0,np.sqrt(1-noise_level)]))
            E_1 = np.array(([0,np.sqrt(noise_level)],[0,0]))
            noisy_state = E_0 @ self.state @ (E_0.conj().T) + E_1 @ self.state @ (E_1.conj().T)
        else:
            raise ValueError("Unknown noise type: %s" % noise_type)

        return(qstate(noisy_state))
    
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

    def __init__(self, mOps, labels=None, tol_positivity=1e-8):
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
        self.n_ops = int(len(self.mOps))
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