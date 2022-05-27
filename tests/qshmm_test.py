import numpy as np
from spqs.qshmm import qsHMM
from spqs.utils import GoldenMean, SNS, A_0p

qGM = qsHMM(GoldenMean(),A_0p)
qSNS = qsHMM(SNS(),A_0p)

def test_qGM_alph():
    assert qGM.alph_size == 2
    assert (qGM.alph == A_0p).all()

def test_q_word():
    np.testing.assert_allclose(qGM.q_word('10').state, \
    [[0.5+0.j, 0. +0.j, 0.5+0.j, 0. +0.j],
       [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
       [0.5+0.j, 0. +0.j, 0.5+0.j, 0. +0.j],
       [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j]])
    np.testing.assert_allclose(qSNS.q_word('01').state, \
    [[0.5+0.j, 0.5+0.j, 0. +0.j, 0. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j, 0. +0.j],
       [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
       [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j]])