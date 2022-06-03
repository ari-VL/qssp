import numpy as np
from spqs.utils import *

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
    assert ket_norm.is_normalized() == True
    assert ket_not_norm.is_normalized() == False

def test_normalize():
    ket_res = np.array([[1/2,0],[0,1/2]])
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

def test_vn_entropy():
    max_mix = qstate(np.array([[1/2, 0],[0, 1/2]]))
    assert ketm.vn_entropy() == 0
    assert max_mix.vn_entropy() == 1
