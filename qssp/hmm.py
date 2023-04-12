import numpy as np
import scipy.linalg as la
from scipy.stats import entropy

def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a-b)<(atol+rtol*np.abs(b))

class HMM:
    '''
    A Hidden Markov Model representation of a process with basic functionality. 
    
    Attributes
    -----------
    Ts : np.array
        Numpy array of labeled transition matrices
    init_dist : np.array
        Array containing the initial probability distribution over hidden states. If None, the stationary state distribution is used
    alphabet : np.array
        Array containing symbols alphabet for HMM
    state_labels : np.array
        Array containing labels for states of HMM

    Methods
    -------
    is_unifilar()
        Bool- returns True if HMM is unifilar
    all_words(L)
        Returns all possible words of length L and their probabilities
    sample_transition(state)
        Samples a possible transition from a given hidden state, with its associated probability. Returnsnext state and the emitted symbol
    sample_words(n, L)
        Samples and returns n words of length L
    stationary_distribution()
        Returns the stationary state distribution of the HMM as a numpy array
    state_entropy(dist=None)
        Returns the Shannon entropy of the given state distribution dist. If None, returns the entropy of the stationary state distribution
    block_entropies(L)
        Returns a list of block entropies for words of length=1 to length=L
    entropy_rate_approx(L)
        Computes and returns the (difference) entropy rate approximation at length L
    excess_entropy_approx(L)
        Returns the Excess Entropy approximation at length L
    '''
    def __init__(self, Ts, init_dist = None):
        '''
        Parameters
        -----------
        Ts: np.array
            Labeled transition matrices of the HMM
        init_dist=None: None or np.array
            Initial state distribution. If none passed, it is taken to be the asymptotic state distribution. 
        '''
        self.Ts = Ts
        self.alphabet  = np.arange(len(Ts))
        self.state_labels = np.arange(len(Ts[0]))
        
        #initialize state distribution as pi if not specified
        if init_dist is None:
            self.init_dist = self.stationary_distribution()
        else:
            self.init_dist = init_dist

    def is_unifilar(self):
        '''bool - returns True if HMM is unifilar'''
        for T in self.Ts:
            #for each labelled transition matrix count number of non zero entries per row and flag false if any is bigger than 0
            non_zero= np.count_nonzero(T, axis=1)
            count= len(non_zero[non_zero>1])
            if count > 0:
                return False

        return True

    def all_words(self, L):
        '''Returns all possible words of length L and their probabilities
        Parameters
        ----------
        L: int
            Length of words to be computed
        Returns
        -------
        tuple
            (words, probs) are lists of allowed words and their associated probabilities
        '''
        init_dist = self.init_dist
        Ts = self.Ts

        if L < 1:
            words = ['']
            probs = [1]
            
            return words, probs

        else:

            words = [str(x) for x in range(len(Ts))]
            probs = []

            for l in range(1,L):
                words_l = []
                for x in range(len(Ts)):
                    for word in words:
                        words_l.append(word + str(x))
                words = words_l

            for word in list(words):
                prob = init_dist.T
                for x in word:
                    prob = np.matmul(prob,Ts[int(x)])
                prob = sum(prob)
                if near(prob,0):
                    words.remove(word)
                else:
                    probs.append(prob)

            return words, probs

    def sample_transition(self, state):
        ''' Samples a possible transition from a given hidden state, with its associated probability. Returnsnext state and the emitted symbol
        Parameters
        -----------
        state: int
            State from which a transition will be sampled
        Returns
        -------
        tuple
            (new_state, symbol) are both int representing the new state and the symbol emitted during transition
        '''
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

    def sample_words(self, n, L):
        '''
        Samples and returns n words of length L
        Parameters
        ----------
        n: int
            Number of words to be sampled and returned
        L: int
            Length of words to be sampled and returned
        Returns
        -------
        list 
            List of sampled words
        '''
        words = []

        #determine initial state for all n words
        init_dist = np.random.choice(np.arange(0,len(self.init_dist)),n,p=self.init_dist)

        for i in range(n):
            word_n = ''

            # set initial state vector
            state = init_dist[i]

            for l in range(L):

                # randomly (with weighted probabilities) choose a transition from current state
                state, symbol = self.sample_transition(state)
                word_n = word_n + symbol

            words.append(word_n)
        return words

    def stationary_distribution(self):
        '''Returns the stationary state distribution of the HMM as a numpy array'''
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
        '''Returns the Shannon entropy of the given state distribution dist. If None, returns the entropy of the stationary state distribution
        Parameters
        ----------
        dist=None : dist
            Distribution whose Shannon entropy will be computed, if None then returns Shannon Entropy of stationary state distribution
        Returns
        -------
        float
            Shannon entropy rate of state distribution
         '''
        #state entropy of the stationary distribution
        if dist is None:
            dist = self.stationary_distribution()
        H = entropy(dist, base=2)
        return H

    def block_entropies(self, L):
        '''Returns a list of block entropies for words of length=0 to length=L
        Parameters
        ----------
        L: int
            Maximum word length for which block entropies will be computed
        Returns
        -------
        list
            List of block entropies for words from length 1 to L
        '''
        block_entropies = [0]

        for l in range(1,L+1):
            block_entropies.append(entropy(self.all_words(l)[1],base=2))

        return block_entropies

    def entropy_rate_approx(self, L):
        '''
        Computes and returns the (difference) entropy rate approximation at length L
        Parameters
        ----------
        L: int
            Length for which entropy rate approximation is computed.
        Returns
        -------
        float
            Entropy rate approximation
        '''

        hmu_L = entropy(self.all_words(L)[1],base=2) - entropy(self.all_words(L-1)[1],base=2)

        return hmu_L

    def excess_entropy_approx(self, L):
        ''' Returns the Excess Entropy approximation at length L
        Parameters
        ----------
        L: int
            Wordlength at which the entropy rate approximation is computed
        Returns
        -------
        float
            Excess entropy approximation
        '''
        block_entropies = self.block_entropies(L)
        #print(L, block_entropies)

        if L > 1:
            EE_L = block_entropies[-1] - L * (block_entropies[-1] - block_entropies[-2])
        else: 
            EE_L = 0

        return EE_L


    def evolve(self, N, init_dist=None, transients=0, word=False):
        '''
        takes an initial mixed state and evolves it N time steps (a single path)

        Parameters
        -----------
        N: int
            Number of time steps to be computed
        init_dist=None: None or np.array
            Initial mixed state
        transients=0: int
            Number of transients to be thrown away from the path.
        word: bool, optional
            If True returns word generated during evolution 

        Returns
        --------
        np.array with the mixed states visited in the path, will have N-transients entries
        '''

        ts = self.Ts
        num_states = len(ts[0])
        mstates = np.zeros((N, num_states))

        if init_dist is None:
            init_dist = self.init_dist

        long_word = self.sample_words(1, N)[0]
        mstates[0]=init_dist

        for i in range(1,len(long_word)):
            mu = np.matmul(mstates[i-1], ts[int(long_word[i])])
            mu = mu/np.sum(mu)
            if not near(np.sum(mu), 1.0):
                raise ValueError("Normalization failed for length-" + str(i) + " mixed state" )
            mstates[i]=mu
        
        mstates = mstates[transients:]
        if word:
            return mstates, long_word
        else:
            return mstates

    def many_paths(self, N, runs, init_dist=None, transients=0):
        #TODO: allow different initial states for every run?
        '''
        Runs evolve 'runs' times and stitches all the resulting mixed states into one array. 
        For any well behaved hmm and a sufficiently large number of N and runs, this should be 
        an approximation of the msp states (w/o transitions)

        Parameters:
        -----------
        N: int
            number of timesteps to evolve initial state
        runs: int
            number of times the evolution is run (starting in the same initial state)
        init_dist: np.array or None
            initial state of hmm
        transients: int
            number of transient states to be ignored for each run of the evolution

        Returns:
        --------
        np.array with the mixed states visited during 'runs' runs of length (N-transients)
        '''
        ms_all = self.evolve(N, init_dist, transients)
        for i in range(runs-1):
            path = self.evolve(N, init_dist, transients)
            ms_all = np.concatenate((ms_all, path),0)
        
        return ms_all