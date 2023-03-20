import numpy as np
from qssp.q_objects import measurement
from qssp.utils import *
import random

def test_init():
    ket3 = np.array([0.5, 0])
    rho3 = np.array([[0.25, 0],[0, 0]])
    state3 = qstate(ket3)
    np.testing.assert_allclose(state3.state, rho3)

def test_addition():
    ket_res = np.array([[1,0],[0,1]])
    np.testing.assert_allclose((ket0 + ket1).state, ket_res)

def test_substraction():
    ket_res = np.array([[0,1],[1,0]])
    np.testing.assert_allclose((ketp - ketm).state, ket_res)

def test_multiplication():
    ket_res = np.array([[2,0],[0,0]])
    np.testing.assert_allclose(2*ket0.state, ket_res)
    np.testing.assert_allclose((ket0 *2).state, ket_res)

def test_div():
    ket_res = np.array([[0,0],[0, 1/3]])
    np.testing.assert_allclose((ket1/3).state, ket_res)

def test_is_normalized():
    ket_norm = qstate(np.array([[1/2,0],[0,1/2]]))
    ket_not_norm = qstate(np.array([[0,3],[0,0]]))
    bloch_ket_norm = bloch_ket(np.pi)
    assert ket_norm.is_normalized() == True
    assert ket_not_norm.is_normalized() == False
    assert bloch_ket_norm.is_normalized() == True

def test_normalize():
    ket_res = np.array([[1/2,0],[0,1/2]])
    ket1.normalize()
    keta = ket0+ket1
    keta.normalize()
    np.testing.assert_allclose(keta.state, ket_res)

def test_is_hermitian():
    not_herm = qstate(np.array([[0, 1j],[0,0]]))
    assert ket0.is_hermitian()==True
    assert not_herm.is_hermitian() == False

def test_is_positive():
    neg_state = qstate(np.array([[2,0],[0,-1]]))
    assert ket1.is_positive() == True
    assert neg_state.is_positive() == False

def test_is_pure():
    max_mix = qstate(np.array([[1/2, 0],[0, 1/2]]))
    assert ketp.is_pure() == True
    assert max_mix.is_pure() == False

def test_is_valid():
    assert ket1.is_valid() == True
    assert (ket1 + ketp).is_valid() == False
    assert (qstate(np.array([[1,1],[0,0]]))).is_valid() == False
    assert (qstate(np.array([[2,0],[0,-1]]))).is_valid() == False

def test_vn_entropy():
    max_mix = qstate(np.array([[1/2, 0],[0, 1/2]]))
    assert ketm.vn_entropy() == 0
    assert max_mix.vn_entropy() == 1

def test_measure():
    np.testing.assert_allclose(ket0.measure(M_01), np.array([1, 0]))
    np.testing.assert_allclose(ket1.measure(M_01), np.array([0, 1]))
    np.testing.assert_allclose(ket0.measure(M_param(np.pi/2)), np.array([0.5, 0.5]))

def test_measure_sample():
    answ = np.array([1, 1, 1, 1, 0, 1, 0])
    np.random.seed(0)
    outcomes = ket0.measure_sample(M_param(np.pi/2), 7)
    assert np.all(outcomes == answ)

def test_m_labels():
    M01 = measurement([M_0, M_1], labels=['a','b'])
    #print (M01.labels)
    assert 'a' in M01.labels
    assert 'b' in M01.labels

def test_m_is_positive():
    M_neg = np.array([[-1,0],[0,0]])
    M_neg1 = measurement([M_neg, M_1])
    assert M_01.is_positive() == True
    assert M_neg1.is_positive() == False

def test_m_is_complete():
    M_inc = measurement([M_0])
    assert M_inc.is_complete() == False
    assert M_01.is_complete() == True

def test_r_mul():
    mul_state = 1/2 * ket0 + 2* ket1
    rmul_state = ket0 * 1/2 + ket1 * 2
    assert mul_state.state.all() == rmul_state.state.all()

def test_noise():
    np.testing.assert_allclose(ketp.add_noise('phaseflip',0.2).state,\
                                np.array([[0.5+0.j, 0.3+0.j],[0.3+0.j, 0.5+0.j]]))
    np.testing.assert_allclose(ket0.add_noise('bitflip',0.2).state,\
                                np.array([[0.8+0.j, 0.0+0.j],[0.0+0.j, 0.2+0.j]]))
    np.testing.assert_allclose((1/2*ket0+1/2*ketp).add_noise('bitphaseflip',0.2).state,\
                                np.array([[0.65+0.j, 0.15+0.j],[0.15+0.j, 0.35+0.j]]))
    np.testing.assert_allclose(ket0.add_noise('depolarizing',0.2).state,\
                                np.array([[0.9+0.j, 0.0+0.j],[0.0+0.j, 0.1+0.j]]))
    np.testing.assert_allclose(ket1.add_noise('amplitude_damping',0.2).state,\
                                np.array([[0.2+0.j, 0.0+0.j],[0.0+0.j, 0.8+0.j]]))