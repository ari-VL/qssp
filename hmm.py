import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a-b)<(atol+rtol*np.abs(b))

class HMM:
    '''
    Ts : np.array,
    list of labled transition matrices

    init : np.array, 
    initial probability distribution over hidden states, if None, stationary state distribution is used
    '''
    def __init__(self, Ts, init = None):
        self.Ts = Ts
        self.init = init
        self.alphabet  = np.arange(len(Ts))
        self.states = np.arange(len(Ts[0]))
        #initialize state distribution as pi
        if init == None:
            self.init = self.stationary_distribution()

    def all_words(self, L):

        init_states = self.init
        Ts = self.Ts

        words = [str(x) for x in range(len(Ts))]
        probs = []

        for l in range(1,L):
            words_l = []
            for x in range(len(Ts)):
                for word in words:
                    words_l.append(word + str(x))
            words = words_l

        for word in list(words):
            prob = init_states.T
            for x in word:
                prob = np.matmul(prob,Ts[int(x)])
            prob = sum(prob)
            if near(prob,0):
                words.remove(word)
            else:
                probs.append(prob)

        return words, probs

    def sample_transition(self, state):
        ''' Samples a possible transition from a given hidden state, with its associated probability.
        Returns next state and the emitted symbol'''
        transition_states = []
        transition_symbols = []
        probs = []

        Ts = self.Ts

        for x in range(len(Ts)):
            for j in range(len(Ts[x][state])):
                #probability of transitioning to state j on symbol x
                prob = Ts[x][state][j]
                if prob > 0:
                    transition_states.append(j)
                    transition_symbols.append(str(x))
                    probs.append(prob)

        #select a transition from the list of all possible transitions based upon probabilities
        new_index = np.random.choice(np.arange(0,len(probs)),p=probs)
        new_state = transition_states[new_index]
        symbol = transition_symbols[new_index]

        return new_state, symbol

    def sample_words(self, n_words, L):
        '''
        Returns n_words sample words of length L
        '''
        words = []

        #determine initial state for all n words
        init_states = np.random.choice(np.arange(0,len(self.init)),n_words,p=self.init)

        for n in range(n_words):
            word_n = ''

            # set initial state vector
            state = init_states[n]

            for l in range(L):

                # randomly (with weighted probabilities) choose a transition from current state
                state, symbol = self.sample_transition(state)
                word_n = word_n + symbol

            words.append(word_n)
        return words

    def stationary_distribution(self):
        #transition matrix of the HMM
        t = np.sum(self.Ts,0)
        t = t.T
        #compute eigenvalues and eigenvector
        d, v = la.eig(t)
        #select eigv with eigv 1
        pi = v[:, near(d,1.0)].T[0]
        #normalize
        pi = np.real(pi/np.sum(pi))
        return pi.T

    def state_entropy(self, dist=None):
        #state entropy of the stationary distribution
        if dist == None:
            dist = self.stationary_distribution()
        H = entropy(dist, base=2)
        return H

    def block_entropies(self, L):
        block_entropies = [0]

        for l in range(1,L):
            block_entropies.append(entropy(self.all_words(l)[1],base=2))

        return block_entropies

    def entropy_rate_approx(self, L):

        hmu_L = entropy(self.all_words(L)[1],base=2) - entropy(self.all_words(L-1)[1],base=2)

        return hmu_L

    def is_unifilar(self):
        for T in self.Ts:
            non_zero= np.count_nonzero(T, axis=1)
            count= len(non_zero[non_zero>1])
            if count > 0:
                return False

        return True

    def excess_entropy_approx(self, L):

        EE_L = entropy(self.all_words(L)[1],base=2) - L * self.entropy_rate_approx(L)

        return EE_L



