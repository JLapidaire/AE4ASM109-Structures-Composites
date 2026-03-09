import numpy as np
import matplotlib.pyplot as plt

def build_S12(E1,E2,G12,v12):
    S12 = np.array([[1/E1 , -v12/E1 , 0],
                    [-v12/E1 , 1/E2 , 0],
                    [0 , 0 , 1/G12]])
    return S12

def build_Q12(E1, E2, G12, v12):
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
    Sxy = np.linalg.inv(T_epsilon) @ S12 @ T_sigma
    return Sxy

def calc_Q_laminate(Q_0, angles,active_plies):
    Q_all = np.zeros((3,3,len(angles)))
    for i in range(len(angles)):
        if active_plies[i] == True :
            T_sigma, T_epsilon = transformation_matrices(angles[i])
            Q_angle = rotate_S(Q_0,T_sigma, T_epsilon)
            Q_all[:,:,i] = Q_angle
        else:
            Q_all[:,:,i] = np.zeros((3,3))
    return Q_all

def calc_ABD(Q,angles,t):
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    T_midplane = t * len(angles) / 2 #define where midplane is
    for k in range(len(angles)):
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

def lamina_strains12(angles,eps_xy_mid,t):
    eps = np.zeros((3,len(angles)))
    for k in range(len(angles)):
        T_sigma, T_epsilon = transformation_matrices(angles[k])
        T_epsilon = np.block([
            [T_epsilon,             np.zeros((3,3))],
            [np.zeros((3,3)), T_epsilon            ]
        ])
        eps12_mid = T_epsilon @ eps_xy_mid
        T_midplane = t * len(angles) / 2
        Z_k = t * k - T_midplane
        if Z_k >= 0 :
            Z_k = t * (k+1) - T_midplane
        eps_ply = [eps12_mid[0] + Z_k * eps12_mid[3],eps12_mid[1] + Z_k * eps12_mid[4], eps12_mid[2] + Z_k * eps12_mid[5]]     
        eps[:,k] = eps_ply
    return eps

def lamina_stress12(eps, Q_all,angles):
    stress = np.zeros((3,len(angles)))
    for i in range(len(angles)):
        stress[:,i] =   Q_all[:,:,i] @ eps[:,i]
    return stress

#patka here
def failure(stress,X_t , X_c ,Y_t , Y_c ,S ):
    F1 = stress[0]/X_t
    F1c = stress[0]/X_c * (-1)
    F2 = stress[1]/Y_t
    F2c = stress[1]/Y_c * (-1)
    Fs = abs(stress[2]/S)
    return any(f > 1 for f in (F1, F1c, F2, F2c, Fs))

#material data
#UD lamina
#all placeholder data
E1 = 140 
E2 = 10
G12 = 5
v12 = 0.3
t = 0.125
#Failure criteria, placeholder values
X_t = 1500
X_c = 1500
Y_t = 1500
Y_c = 1500
S = 1500
#laminate data, also placeholder
angles = [0,45,-45,90]

active_plies = []
for i in range(len(angles)):
    active_plies.append(True)

S12 = build_S12(E1,E2,G12,v12)
Q_0 = build_Q12(E1,E2,G12,v12)
Q_all = calc_Q_laminate(Q_0,angles,active_plies)
ABD = calc_ABD(Q_all,angles,t)
print(ABD)

'''
F = [0,0,0,0,0,0] #N and M vector, placeholder values
print(active_plies)
F_increment = np.array([0.1,0,0,0,0,0])
ABD = calc_ABD(Q_all,angles,t)
abd = np.linalg.inv(ABD)
while any(active_plies):
    eps_xy_mid = abd @ F
    eps = lamina_strains12(angles,eps_xy_mid,t)
    stress = lamina_stress12(eps, Q_all,angles)
    for i in range(len(angles)):
        failed_ply = failure(stress[:,i],X_t , X_c ,Y_t , Y_c ,S )
        if failed_ply == True:
            active_plies[i]= False
            Q_all = calc_Q_laminate(Q_0,angles,active_plies)
            ABD = calc_ABD(Q_all,angles,t)
            abd = np.linalg.inv(ABD)
    F += F_increment

print("all plies have failed, when the applied load was:",F)
'''