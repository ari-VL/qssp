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