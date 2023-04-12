import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

from .hmm import HMM
from .q_objects import qstate

def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a-b)<(atol+rtol*np.abs(b))

class qsHMM:
    '''
    A quantum state Hidden Markov Model consisting of a classical HMM
    and an alphabet of quantum states.
    
    Attributes
    -----------
    HMM : HMM object
        Numpy array of labeled transition matrices
    alph : list of qstates
        List of quantum states which are associated with symbols of the classical HMM
    alph_size : int
        Size of the alphabet of quantum states

    Methods
    -------
    q_word(word)
        Turns given classical word into quantum state using the quantum alphabet
    q_words(n, L)
        Samples words of length L with appropriate probabilities n times and returns list associated q_words
    q_block(L, join = True)
        Returns single quantum state represetning all words of length L
        or (if join == False) a quantum state and probability for each word
    q_block_entropies(L)
        Returns a list of von Neumann entropies for blocks of length=1 to length=L
    q_entropy_rate(L)
        Computes and returns the (difference) von Neumann entropy rate approximation at length L
    q_excess_entropy(L)
        Returns the quantum excess entropy approximation at length L
    get_measured_machine(measurement)
        Returns the classical HMM generator corresponding to a particular measurement basis
    observer_state_uncertainty(measurement, L, ever_synched = False)
        Returns average state uncertainty of observer measuring in fixed basis on up to L sites
    '''

    def __init__(self, model, alph, noise_type='None',noise_level=0):
        #should it take in Trans Matrices?
        if not isinstance(model, HMM):
            raise TypeError("Model must be specified as HMM object")
        self.HMM=model
        self.noise_type=noise_type
        self.noise_level=noise_level
        if noise_level != 0:
            self.alph = []
            for alph_state in alph:
                self.alph.append(alph_state.add_noise(noise_type, noise_level))
        else:
            self.alph= alph
        self.alph_size= int(len(alph))
        #check alph size is equal to no. of Ts
        
    def q_word(self, word):
        #takes in a classical word and turns it into a quantum state
        q_word = self.alph[int(word[0])].state
        for i in range(1,len(word)):
            q_word = np.kron(q_word, self.alph[int(word[i])].state)
        q_word = qstate(q_word)
        return q_word
        
    def q_words(self, n, L):
        #samples n words of length L and returns corresponding quantum states 
        if not isinstance(L, int):
            raise TypeError("Invalid word length (must be non-negative integer).")
        elif (L <= 0):
            raise ValueError("L must be greater than 0.")
        
        words = self.HMM.sample_words(n,L)
        q_words = [self.q_word(word) for word in words]
        return q_words
        
    def q_block(self, L, join=True):
        #calculates all the words of length L and their probs and returns the joint quantum state
        if not isinstance(L, int):
            raise TypeError("Invalid block length (must be non-negative integer).")
        elif (L <= 0):
            raise ValueError("L must greater than 0.")
        
        words, c_probs = self.HMM.all_words(L)
        if join:
            q_block = c_probs[0] * self.q_word(words[0]).state
            for i in range(1,len(words)):
                q_block = q_block + c_probs[i]*self.q_word(words[i]).state
            q_block = qstate(q_block)
            return q_block
        else: 
            q_seq = []
            for i in range(len(words)):
                q_seq.append(self.q_word(words[i]).state)
            return q_seq, c_probs

    def q_block_entropies(self, L):
        if not isinstance(L, int):
            raise TypeError("Invalid block length (must be non-negative integer).")
        elif (L <= 0):
            raise ValueError("L must be greater than 0.")
        
        ents = [0] + [self.q_block(l).vn_entropy() for l in range(1,L+1)]
        return ents
        
    def q_entropy_rate(self, L):
        if not isinstance(L, int):
            raise TypeError("Invalid block length (must be non-negative integer).")
        elif (L < 1):
            raise ValueError("L must be greater than 0.")
        
        q_ents = self.q_block_entropies(L)
        s_est = q_ents[-1] - q_ents[-2]
        return s_est
        
    def q_excess_entropy(self, L):
        if not isinstance(L, int):
            raise TypeError("Invalid block length (must be non-negative integer).")
        elif (L < 1):
            raise ValueError("L must be greater than 0.")
        
        EE_est = self.q_block(L).vn_entropy() - L * self.q_entropy_rate(L)
        return EE_est

    def get_measured_machine(self, measurement):
        '''
        Computes the measured machine when applying "measurement" to the qs_HMM, returns an HMM object
        '''
        #size of the classical alphabet of the measured machine
        c_abet_size = len(measurement.mOps)
        #number of states in the HMM
        n_states = self.HMM.Ts[0].shape[0]

        #initialize an array that will store the labeled transition matrices for the measured machine
        measured_Ts = np.zeros([c_abet_size, n_states, n_states])
        
        #each iteration over i computes th labeled transition matrix of the measured machine T_i
        for i in range(c_abet_size):
            #Initialize labeled transition matrix
            t = np.zeros([n_states, n_states])
            for j in range(self.alph_size):
                #compute the probability of measurement outcome i when measuring quantum state j
                prob = self.alph[j].measure(measurement)[i]
                #weigh the quantum labeled transition matrix of state j by the probability of seeing out come i, and add that for all js
                t += (self.HMM.Ts[j]*prob)
            #store labeled transition matrix T_i    
            measured_Ts[i] = t
        #create HMM object with the correct transition matrices
        measured_mach = HMM(measured_Ts)
        return measured_mach

    def observer_state_uncertainty(self,measurement,L,ever_synched=False):
        '''
        Returns average state uncertainty of observer measuring in fixed basis on up to L sites
        '''

        S = [[self.HMM.init]]
        probs = [[1]]
        H_avg = [entropy(S[0][0],base=2)]
        
        meas = self.get_measured_machine(measurement)
        Ts = meas.Ts
        
        for l in range(1,L+1):
            S_t = []
            probs_t = []
            H_avg_t = 0

            for s in range(len(S[-1])):
                # Calculates mixed states given L observations using mixed states given (L-1) observations
                S_prev = S[-1][s]
                prob_prev = probs[-1][s]

                if(ever_synched and self.is_synched(S_prev)):
                    # If 'ever_synched', stop calculating new mixed states after synchronizing observation
                    S_t.append(S_prev)
                    probs_t.append(prob_prev)
                
                else:
                    for i in range(len(meas.Ts)):
                        S_new = [0] * len(meas.Ts[0])

                        for k in range(len(meas.Ts[0])):
                            prob_k = 0

                            for j in range(len(meas.Ts[0])):
                                if not near(Ts[i][j][k],0):
                                    # Calculate probability of each new mixed state
                                    prob_k += S_prev[j]* Ts[i][j][k]

                            S_new[k] += prob_k

                        if not near(sum(S_new),0):
                            S_t.append(S_new/sum(S_new))
                            new_prob = prob_prev*sum(S_new)
                            probs_t.append(new_prob)
                            H_avg_t += new_prob * entropy(S_new/sum(S_new),base=2)

            S.append(S_t)
            probs.append(probs_t)
            H_avg.append(H_avg_t)

        # Returns list of entropies over mixed states for different L values
        return H_avg

    def is_synched(self, mixed_state):
        # If mixed state is concentrated in one generator state returns True, else False
        if not isinstance(mixed_state, list):
            raise TypeError("Mixed state must be list of probabilities")
        elif not len(mixed_state) == len(self.HMM.stationary_distribution()):
            raise ValueError("Mixed state has the wrong number of elements.")
        elif not near(sum(mixed_state),1):
            raise ValueError("Mixed state probabilities must sum to 1.")

        for s in mixed_state:
            if (not near(s,0) and not near(s,1)):
                return False
        
        return True