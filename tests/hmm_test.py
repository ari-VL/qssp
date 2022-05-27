import numpy as np
from spqs.hmm import HMM
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

