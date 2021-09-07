'''
Class "HMM_SPQS", initialized by (Transition Matrices, Alphabet)

Methods: 
    all_words(L): all words of length L with associated probabilities
    sample(L): sample word of length L according to probabilities

    quantum_block_entropies(L)
    quantum_entropy_rate(L)
    quantum_excess_entropy()

    get_generator() --> generator, same transition matrices, unifilar, with some classical alphabet
    get_measured_machine(measurement) --> nonuni, transduced transition matrices

    [maybe]
    (get_generator) or
    (generator_entropy_rate)
    generator_statistical_complexity) 

    [future]
    adaptive_measurement
    nonlocal_measurement


    Class "HMM" (transition_matrices, symbols)

    Methods: 

    entropy_rate
    msp_construction
    others
'''

import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

import hmm
import q_utils

class qsHMM:

    def __init__(self, HMM, alph):
        #should it take in Trans Matrices?
        self.HMM=HMM
        self.alph= alph
        self.alph_size= len(alph)
        #check alph size is equal to no. of Ts

    def q_word(self, c_word,L):
        #takes in a classical word and turns it into a quntum state word
        q_word = self.alph[int(c_word[0])].state
        for i in range(1,len(c_word)):
            q_word = np.kron(q_word, self.alph[int(c_word[i])].state)
        q_word = qstate(q_word)
        return q_word
        
    def q_words(self, n_words, L):
        #samples n_words of length L 
        c_words = self.HMM.sample_words(n_words,L)
        q_words = [self.q_word(c_word, L) for c_word in c_words]
        return q_words
        
    def q_block(self, L, join=True):
        #calculates all the words of length L and their probs and returns the joint quantum state
        #TODO: test 
        c_words, c_probs = self.HMM.all_words(L)
        if join:
            q_block = c_probs[0] * self.q_word(c_words[0],L).state
            for i in range(1,len(c_words)):
                q_block = q_block + c_probs[i]*self.q_word(c_words[i],L).state
            q_block = qstate(q_block)
            return q_block
        else: 
            q_seq = []
            for i in range(len(c_words)):
                q_seq.append(self.q_word(c_words[i],L).state)
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