import numpy as np
import matplotlib.pyplot as plt

def build_S12(E1,E2,G12,v12):
    S12 = np.array([[1/E1 , -v12/E1 , 0],
                    [-v12/E1 , 1/E2 , 0],
                    [0 , 0 , 1/G12]])
    return S12

def build_Q12(E1, E2, G12, v12):
    v21 = v12 * E2 / E1
    denom = 1.0 - v12 * v21
    Q12 = np.array([
        [E1 / denom,      v12 * E2 / denom, 0.0],
        [v12 * E2 / denom, E2 / denom,      0.0],
        [0.0,             0.0,              G12]
    ])
    return Q12

def transformation_matrices(theta):

    m = np.cos(np.deg2rad(theta))
    n = np.sin(np.deg2rad(theta))

    T_sigma = np.array([
        [ m**2,  n**2,   2*m*n],
        [ n**2,  m**2,  -2*m*n],
        [-m*n,   m*n,   m**2 - n**2]
    ])

    T_epsilon = np.array([
        [ m**2,  n**2,   m*n],
        [ n**2,  m**2,  -m*n],
        [-2*m*n, 2*m*n,  m**2 - n**2]
    ])

    return T_sigma, T_epsilon

def rotate_S(S12,T_sigma, T_epsilon):
    #is called rotate S but work for any 3x3 matrix rotation
    Sxy = np.linalg.inv(T_epsilon) @ S12 @ T_sigma
    return Sxy

def calc_Q_laminate(Q_0, angles,active_plies, failed_once,n_plies):
    #The function creates a 3 by 3 by num of plies array of Q matricies for a laminate made out of identical plies
    #basicaly Q_all[:,:,i] is the Qxy matrix in global coordinates for ply number i counting from the bottom 
    Q_all = np.zeros((3,3,n_plies))
    for i in range(n_plies):
        #this part checks if the given ply is still active (true) or has failed (false)
        #and if it is active, was it the first or second failure
        if active_plies[i] == True :
            if failed_once[i] == True :
                T_sigma, T_epsilon = transformation_matrices(angles[i])
                Q_0degraded = Q_0
                Q_0degraded[1,1] *= 0.15
                Q_0degraded[0,1] *= 0.15
                Q_0degraded[1,0] *= 0.15
                Q_0degraded[2,2] *= 0.15
                '''Im not 100% sure if i understood the degradation rule correctly. what i did is equivalent to changing E2new = 0.15 E2.'''
                '''Im not sure if v21 should be degraded as well'''
                Q_angle = rotate_S(Q_0degraded,T_sigma, T_epsilon)
                Q_all[:,:,i] = Q_angle
            else :
                T_sigma, T_epsilon = transformation_matrices(angles[i])
                Q_angle = rotate_S(Q_0,T_sigma, T_epsilon)
                Q_all[:,:,i] = Q_angle
        else:
            Q_all[:,:,i] = np.zeros((3,3))
    return Q_all

def calc_ABD(Q,t,n_plies):
    #basing on the Q_all calculates ABD matrix. Q_all has to be 3 x 3 x number of plies
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    T_midplane = t * n_plies / 2 #define where midplane is
    for k in range(n_plies):
        for i in range(3):
            for j in range(3):
                Z_k = t * k - T_midplane
                Z_k_next = t * (k+1) - T_midplane
                A[i,j] += Q[i,j,k] * (Z_k_next - Z_k)
                B[i,j] += Q[i,j,k] * (Z_k_next**2 - Z_k**2)/2
                D[i,j] += Q[i,j,k] * (Z_k_next**3 - Z_k**3)/3
    AB = np.vstack((A,B))
    BD = np.vstack((B,D))
    ABD = np.hstack((AB,BD))
    return ABD

def calc_symmetric_A(Q,t,n_plies):
    '''idea to make the code run faster: vectorize it instead of nested loops and use z as linspace'''
    #laminates in task 2 and 3 are symetric, so B is equal to 0. the code will run faster if we decouple A and D
    #basing on the Q_all calculates A matrix. Q_all has to be 3 x 3 x number of plies
    A = np.zeros((3,3))
    T_midplane = t * n_plies / 2 #define where midplane is
    half =int(n_plies)/2
    for k in range(half): #we can calculate half of A and multiply it times 2
        for i in range(3):
            for j in range(3):
                Z_k = t * k - T_midplane
                Z_k_next = t * (k+1) - T_midplane
                A[i,j] += Q[i,j,k] * (Z_k_next - Z_k)
    A *= 2
    return A


def calc_symmetric_D(Q,t,n_plies):
    '''idea to make the code run faster: vectorize it instead of nested loops and use z as linspace'''
    #laminates in task 2 and 3 are symetric, so B is equal to 0. the code will run faster if we decouple A and D
    #basing on the Q_all calculates D matrix. Q_all has to be 3 x 3 x number of plies
    D = np.zeros((3,3))
    T_midplane = t * n_plies / 2 #define where midplane is
    half = int(n_plies/2)
    for k in range(half): #we can calculate half of A and multiply it times 2
        for i in range(3):
            for j in range(3):
                Z_k = t * k - T_midplane
                Z_k_next = t * (k+1) - T_midplane
                D[i,j] += Q[i,j,k] * (Z_k_next**3 - Z_k**3)/3
    D *= 2
    return D

def lamina_strains12(angles,eps_xy_mid,t,n_plies):
    #converts the strain in 12 direction in midplane to strains in each ply IN LOCAL (PLY) COORDINATE SYSTEMS.
    #returns 3 x number of plies matrix of strains for all plies. 
    eps = np.zeros((3,n_plies))
    for k in range(n_plies):
        T_sigma, T_epsilon = transformation_matrices(angles[k])
        T_epsilon = np.block([
            [T_epsilon,np.zeros((3,3))],
            [np.zeros((3,3)), T_epsilon]
        ])
        eps12_mid = T_epsilon @ eps_xy_mid
        T_midplane = t * n_plies / 2
        Z_k = t * k - T_midplane
        ''' here i check if the strain is higher on the bottom or on top of the plane and choose the higher values
        left blank until im sure what should i take (bottom top or both)'''







        if Z_k >= 0 :
            Z_k = t * (k+1) - T_midplane
        eps_ply = [eps12_mid[0] + Z_k * eps12_mid[3],eps12_mid[1] + Z_k * eps12_mid[4], eps12_mid[2] + Z_k * eps12_mid[5]]     
        eps[:,k] = eps_ply
    return eps

def lamina_stress12(eps, Q_all,n_plies):
    stress = np.zeros((3,n_plies))
    for i in range(n_plies):
        stress[:,i] =   Q_all[:,:,i] @ eps[:,i]
    return stress

'''patka here'''
def failure(stress,X_t , X_c ,Y_t , Y_c ,S ):
    '''it is important that this function returns if the ply has failed and WHICH FAILURE MODE'''
    F1 = stress[0]/X_t
    F1c = stress[0]/X_c * (-1)
    F2 = stress[1]/Y_t
    F2c = stress[1]/Y_c * (-1)
    Fs = abs(stress[2]/S)
    return any(f > 1 for f in (F1, F1c, F2, F2c, Fs))

#material data
#UD lamina
E1 = 172.3*10**3 #MPa 
E2 = 10.2*10**3 #MPa
G12 = 5.58*10**3 #MPa
v12 = 0.25 #[-]
'''thickness is still a placeholder value'''
t = 0.125 #mm 
#Failure criteria
X_t = 1923 #MPa
X_c = 1480 #MPa
Y_t = 84 #MPa
Y_c = 220 #MPa
S = 144.5 #MPa

base = [0,45,-45,90,30]
angles = (base*3)+(base*3)[::-1]
n_plies = int(len(angles)) #total number of plies

active_plies = []
for i in range(n_plies):
    active_plies.append(True)

has_failed_once = []
for i in range(n_plies):
    has_failed_once.append(False)

S12 = build_S12(E1,E2,G12,v12)
Q_0 = build_Q12(E1,E2,G12,v12)
Q_all = calc_Q_laminate(Q_0,angles,active_plies,has_failed_once,n_plies)


ABD = calc_ABD(Q_all,angles,t,n_plies)
print(ABD)

number = 10 #number of ratios that will be used during analysis 
ratios = np.zeros((2,number))
for i in range(number):
    #in analysis we get maximum of 2 non zero loads (Nx, Ny or whatever)
    #in this way we get evenly spaced set of load ratios to create failure envelope
    angle = 2 *np.pi / number
    ratios(0,i) = np.cos(angle) 
    ratios(1,i) = np.sin(angle)

#list of values when first and last plies at each load ratio
first_ply_failure_NxNy = np.zeros(2,number)
last_ply_failure_NxNy = np.zeros(2,number)
first_ply_failure_NyNs = np.zeros(2,number)
last_ply_failure_NyNs = np.zeros(2,number)

'''
F = [0,0,0,0,0,0] #N and M vector, placeholder values
print(active_plies)
F_increment = np.array([0.1,0,0,0,0,0])
ABD = calc_ABD(Q_all,t,n_plies)
abd = np.linalg.inv(ABD)
while any(active_plies):
    eps_xy_mid = abd @ F
    eps = lamina_strains12(angles,eps_xy_mid,t,n_plies)
    stress = lamina_stress12(eps, Q_all,n_plies)
    for i in range(n_plies):
        failed_ply = failure(stress[:,i],X_t , X_c ,Y_t , Y_c ,S )
        if failed_ply == True:
            active_plies[i]= False
            Q_all = calc_Q_laminate(Q_0,angles,active_plies,has_failed_once,n_plies)
            ABD = calc_ABD(Q_all,t,n_plies)
            abd = np.linalg.inv(ABD)
    F += F_increment

print("all plies have failed, when the applied load was:",F)
'''