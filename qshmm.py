import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

from .hmm import HMM
from .q_utils import qstate

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
        Turns given classical word of into quantum state using the quantum alphabet
    q_words(n, L)
        Samples words of length L with appropriate probabilities n times and returns list associated q_words
    q_block(L, join = True)
        Returns single quantum state represetning all words of length L
        or (if join == False) a quantum state and probability for each word
    q_block_entropies(L):
        Returns a list of von Neumann entropies for blocks of length=1 to length=L
    q_entropy_rate(L)
        Computes and returns the (difference) von Neumann entropy rate approximation at length L
    q_excess_entropy(L)
        Returns the quantum excess entropy approximation at length L
    get_measured_machine(measurement)
        Returns the classical HMM generator corresponding to a particular measurement basis
    '''

    def __init__(self, HMM, alph):
        #should it take in Trans Matrices?
        self.HMM=HMM
        self.alph= alph
        self.alph_size= len(alph)
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
        words = self.HMM.sample_words(n,L)
        q_words = [self.q_word(word) for word in words]
        return q_words
        
    def q_block(self, L, join=True):
        #calculates all the words of length L and their probs and returns the joint quantum state
        #TODO: test 
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
        if L > 0:
            ents = [self.q_block(l).vn_entropy() for l in range(1,L+1)]
        else:
            ents = []
            print("ERROR: Invalid block length (must be a positive integer)")
        return ents
        
    def q_entropy_rate(self, L):
        if L > 1:
            s_est = self.q_block(L).vn_entropy() - self.q_block(L-1).vn_entropy()
        else:
            s_est = 0
            print("ERROR: Invalid block length (must be greater than 1)")
        return s_est
        
    def q_excess_entropy(self, L):
        if L > 1:
            EE_est = self.q_block(L).vn_entropy() - L * self.q_entropy_rate(L)
        else:
            EE_est = 0
            print("ERROR: Invalid block length (must be greater than 1)")
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