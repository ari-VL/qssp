import numpy as np
from spqs.utils import GoldenMean, SNS, Nemo

def test_GM_unifilar():
    assert GoldenMean().is_unifilar() == True

def test_SNS_unifilar():
    assert SNS().is_unifilar() == False

def test_Nemo_unifilar():
    assert Nemo().is_unifilar() == True

def test_GM_all_words():
    assert GoldenMean().all_words(3) == (['010','110','101','011','111'],[1/6,1/6,1/3,1/6,1/6])

def test_SNS_all_words():
    assert SNS().all_words(3) == (['000', '100', '010', '001', '101'],
 [0.3125, 0.18749999999999997, 0.25, 0.1875, 0.06249999999999999])

def test_GM_sample_words():
    all_words = GoldenMean().all_words(3)
    for word in GoldenMean().sample_words(5,3):
        assert word in all_words[0]

def test_SNS_sample_words():
    all_words = SNS().all_words(3)
    for word in SNS().sample_words(5,3):
        assert word in all_words[0]

def test_GM_stationary_dist():
    np.testing.assert_allclose(GoldenMean().stationary_distribution(), np.array([2/3,1/3]))

def test_SNS_stationary_dist():
    np.testing.assert_allclose(SNS().stationary_distribution(), np.array([1/2, 1/2]))

def test_GM_stationary_entropy():
    assert GoldenMean().state_entropy() == 0.9182958340544894
    assert GoldenMean().state_entropy([1,0]) == 0

def test_SNS_stationary_entropy():
    assert SNS().state_entropy == 1
    assert SNS().state_entropy([0,1])== 0

def test_GM_sample_transition():
    A_transitions = [(0,'1'),(1,'0')]
    B_transitions = [(0,'1')]
    sampled_A_transitions = [GoldenMean().sample_transition(0) for i in range(10)]
    sampled_B_transitions = [GoldenMean().sample_transition(1) for i in range(10)]

    for i in range(10):
        assert sampled_A_transitions[i] in A_transitions
        assert sampled_B_transitions[i] in B_transitions

def test_GM_block_entropies():
    np.testing.assert_allclose(GoldenMean().block_entropies(10), [0, 0.9182958340544894, 1.584962500721156, 2.251629167387823, 2.91829583405449, 3.584962500721157, 4.251629167387824, 4.91829583405449, 5.584962500721157, 6.251629167387822, 6.918295834054489])

def test_SNS_block_entropies():
    np.testing.assert_allclose(SNS().block_entropies(10), [0, 0.8112781244591328, 1.5, 2.1800365325772657, 2.8584585933443494, 3.5364882355820626, 4.2144073285182, 4.892292155061646, 5.570165618160434, 6.248035117839492, 6.925903179808621])

def test_GM_entropy_rate_approx():
    assert GoldenMean().entropy_rate_approx(1) == 0.9182958340544894
    assert GoldenMean().entropy_rate_approx(10) == 0.666666666666667

def test_SNS_entropy_rate_approx():
    assert SNS().entropy_rate_approx(1) == 0.8112781244591328
    assert SNS().entropy_rate_approx(10) == 0.6778680619691295

def test_GM_excess_entropy_approx():
    assert GoldenMean().excess_entropy_approx(1) == 0
    assert GoldenMean().excess_entropy_approx(5) == 0.25162916738782215
    assert GoldenMean().excess_entropy_approx(10) == 0.2516291673878195

def test_SNS_excess_entropy_approx():
    assert SNS().excess_entropy_approx(5) == 0.14634002439349647
    assert SNS().excess_entropy_approx(10) == 0.14722256011732604