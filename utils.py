import numpy as np
from .hmm import HMM
from .q_objects import qstate, measurement
# Some useful Machines 

#Two state machines

def GoldenMean(p=0.5,init_dist=None): 
    t0 = np.array([[0, p],[0,0]])
    t1 = np.array([[1-p, 0],[1,0]])
    return HMM([t0,t1],init_dist)

def Even(p=0.5,init_dist=None):
    t0 = np.array([[1-p, 0],[0,0]])
    t1 = np.array([[0, p],[1,0]])
    return HMM([t0,t1],init_dist)

def SNS(p=0.5,q=0.5,init_dist=None):
    t0 = np.array([[p, 1-p],[0,q]])
    t1 = np.array([[0, 0],[1-q,0]])
    return HMM([t0,t1],init_dist)


#Three State Machines
def Nemo(p=0.5, q=0.5,init_dist=None):
    t0 = np.array([[p,0,0],[0,0,0],[q,0,0]])
    t1 = np.array([[0,1-p,0],[0,0,1],[1-q,0,0]])
    return HMM([t0,t1],init_dist)

def RIP(p=0.5,q=0.5,init_dist=None):
    t0 = np.array([[0,p,0],[0,0,q],[0,0,0]])
    t1 = np.array([[0,0,1-p],[0,0,1-q],[1,0,0]])
    return HMM([t0,t1],init_dist)


# #Some useful states 
identity = qstate(np.array([[1,0],[0,1]]), test_norm=False)
sigma_x = qstate(np.array([[0,1],[1,0]]), test_norm=False)
sigma_y = qstate(np.array([[0,-1j],[1j,0]]), test_norm=False)
sigma_z = qstate(np.array([[1,0],[0,-1]]), test_norm=False)

ket0a = np.array([1,0])
ket1a = np.array([0,1])
ketpa = (1/np.sqrt(2))*(ket0a+ket1a)
ketma = (1/np.sqrt(2))*(ket0a-ket1a)

def bloch_ket(theta, phi=0):
    a = np.cos(theta/2)
    b = np.sin(theta/2)
    ket = a*ket0a + b*np.exp(1j*phi)*ket1a
    return qstate(ket)

ket0 = qstate(ket0a)
ket1 = qstate(ket1a)
ketp = qstate(ketpa)
ketm = qstate(ketma)

A_01 = np.array([ket0,ket1])
A_0p = np.array([ket0,ketp])
A_pm = np.array([ketp,ketm])

M_0 = np.array([[1,0],[0,0]])
M_1 = np.array([[0,0],[0,1]])
M_01 = measurement([M_0,M_1])

def M_param(theta, phi=0):
    a = np.cos(theta/2)
    b = np.sin(theta/2)
    ket = a*ket0a + b*np.exp(1j*phi)*ket1a
    M0 = np.outer(ket.conj().T, ket)
    M1 = np.identity(2) - M0
    return measurement([M0,M1])