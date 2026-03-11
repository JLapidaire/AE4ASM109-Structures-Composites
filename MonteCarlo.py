import numpy as np
from assignment1 import failure


def RandomVariable(Mean,STD):
    return np.random.normal(loc=Mean,scale=STD)

def GenerateProperties():
    E1 = RandomVariable(172.3,9.28) #GPa
    E2 = RandomVariable(10.2,3.58) #GPa
    v12 = RandomVariable(0.25,0.018) #-
    G12 = RandomVariable(5.58,1.97) #GPa
    Xt = RandomVariable(1923,188.3) #MPa
    Yt = RandomVariable(84,8.2) #MPa
    Xc = 1480 #MPa
    Yc = 220 #MPa (Not Distributed?)
    S = RandomVariable(144.5,7.33) #MPa

    Properties = np.array([E1,E2,v12,G12,Xt,Yt,Xc,Yc,S])

    return Properties

def RunSimulation():
    x = 0
    Bool = 0
    Plies = [0,90,45,-45,-45,45,90,0,0,90,45,-45,-45,45,90,0]
    N = 783 #N/mm
    Loading = [np.cos(np.radians(45))*N,np.sin(np.radians(45))*N,0,0,0,0]
    while Bool == 0:
        x += 1
        Properties = GenerateProperties()

        #Needed: Failure calculation with input, properties and laminate (defined here), and output boolean for FPF
        Bool = failure(Properties,Plies,Loading)

    return x

def Convergence(R):
    P = np.zeros(R)
    for i in range(R):
        n = RunSimulation()
        P[i] += (1/n)

    Pf = np.sum(P) / R
    return Pf

print(GenerateProperties())