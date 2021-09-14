import numpy as np
from q_utils import qstate, measurement
# Some useful Machines 

#Two state machines

def GoldenMean(p=0.5): 
    t0 = np.array([[0, p],[0,0]])
    t1 = np.array([[1-p, 0],[1,0]])
    return np.array([t0,t1])

def SNS(p=0.5,q=0.5):
    t0 = np.array([[p, 1-p],[0,q]])
    t1 = np.array([[0, 0],[1-q,0]])
    return np.array([t0,t1])


#Three State Machines
def Nemo(p=0.5, q=0.5):
    t0 = np.array([[p,0,0],[0,0,0],[q,0,0]])
    t1 = np.array([[0,1-p,0],[0,0,1],[1-q,0,0]])
    return np.array([t0,t1])

def RIP(p=0.5,q=0.5):
    t0 = np.array([[0,p,0],[0,0,q],[0,0,0]])
    t1 = np.array([[0,0,1-p],[0,0,1-q],[1,0,0]])
    return np.array([t0,t1])


# #Some useful states 
ket0a = np.array([1,0])
ket1a = np.array([0,1])
ketpa = (1/np.sqrt(2))*(ket0+ket1)
ketma = (1/np.sqrt(2))*(ket0-ket1)

def bloch_ket(theta, phi=0):
    a = np.cos(theta/2)
    b = np.sin(theta/2)
    ket = a*ket0a + b*np.exp(i*phi)*ket1a
    return qstate(ket)

ket0 = qstate(ket0a)
ket1 = qstate(ket1a)
ketp = qstate(ketpa)
ketm = qstate(ketma)