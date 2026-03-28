import numpy as np

#######################
#   strain = S stress
#   stress = C strain

### UD LAMINA ###

### IN PLANE
def inplane_matrices(E1, E2, v12, G12):
    # Finds in S and Q matrices for on-axis lamina
    v21 = E2/E1 * v12

    S = np.array([[1/E1,     -v21/E2,    0],
                 [-v12/E1,  1/E2,        0],
                 [0,             0,      1/G12]])

    Q = np.linalg.inv(S)
    return S, Q

### ROTATION MATRICES
def transformation_matrices(theta):
    # Finds transformation for off-axis lamina
    m = np.cos(theta)
    n = np.sin(theta)
    T_sigma =  np.array([[m**2,    n**2,  2*m*n],
                        [n**2,      m**2,  -2*m*n],
                        [-m*n,      m*n,   (m**2 - n**2)]])

    T_epsilon = np.array([[m**2,    n**2,   m*n],
                        [n**2,     m**2,   -m*n],
                        [-2*m*n,   2*m*n,  (m**2 - n**2)]])
    return T_sigma, T_epsilon

def offaxis_matrices(Q, theta):
    # Rotates Q matrix into local fibre orientations
    T_sigma, T_epsilon = transformation_matrices(theta)
    # S_theta = np.linalg.inv(T_epsilon) @ S @ T_sigma
    Q_theta = np.linalg.inv(T_sigma) @ Q @ T_epsilon
    return Q_theta

def calc_ABD(Q_all, t, n_plies):
    # Inputs:   Array consisting of Q for each ply in fibre orientations,
    #           thickness of each ply (assumes the same for all) in mm,
    #           number of plies
    # Q_all has the size of 3 x 3 x number of plies
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    T_midplane = t * n_plies / 2 # defines where midplane is
    for k in range(n_plies):
        z_k = t * k - T_midplane
        z_k_next = t * (k+1) - T_midplane
        A[:,:] += Q_all[:,:,k] * (z_k_next - z_k)
        B[:,:] += Q_all[:,:,k] * (z_k_next**2 - z_k**2)/2
        D[:,:] += Q_all[:,:,k] * (z_k_next**3 - z_k**3)/3
    ABD = np.zeros((6,6))
    ABD[0:3, 0:3] = A[:, :]
    ABD[0:3, 3:6] = B[:, :]
    ABD[3:6, 0:3] = B[:, :]
    ABD[3:6, 3:6] = D[:, :]
    return ABD

def calc_Q_laminate(Q_0, Q_0degraded, angles, active_plies, failed_once, n_plies):
    # The function creates a 3 x 3 x num of plies array of Q matrices for a laminate made out of identical plies
    # basically Q_all[:,:,i] is the Qxy matrix in global coordinates for ply number i counting from the bottom
    # Inputs: Q_0 array per initial ply at 0 deg;
    #         Q_0degraded per degraded ply at 0 deg;
    #         array with angle orientation per ply;
    #         true/false array stating whether a ply is able to carry loads (by default all true);
    #         true/false array stating whether a ply failed in transverse direction only
    #                           (0.15 degradation rule is applied) ((by default all false));
    #         number of plies;
    # Output: 3 x 3 x num of plies array of Q matrices for each ply in the entire laminate in global coordinates
    Q_all = np.zeros((3,3,n_plies))
    for i in range(n_plies):
        #this part checks if the given ply is still active (true) or has failed (false)
        #and if it is active, was it the first or second failure
        if active_plies[i] == True :
            if failed_once[i] == True :
                Q_angle = offaxis_matrices(Q_0degraded, angles[i])
                Q_all[:,:,i] = Q_angle
            else :
                Q_angle = offaxis_matrices(Q_0, angles[i])
                Q_all[:,:,i] = Q_angle
        else:
            Q_all[:,:,i] = np.zeros((3,3))
    return Q_all


def lamina_strains12(angles, abd, F, t, n_plies, active_plies):
    # converts the strain in 12 direction in midplane to strains in each ply IN LOCAL (PLY) COORDINATE SYSTEMS.
    # returns 3 x number of plies matrix of strains for all plies in local ply coordinates
    # Inputs: array with orientation angles of each ply
    #         inverted ABD matrix
    #         6x1 force intensity vector F in N/mm
    #         thickness of the plies
    #         number of plies
    # Outputs: 2 x 3 x n_plies array with 1, 2, and 12 strain values per top and bottom of each ply

    strain_xy_mid = abd @ F
    strain = np.zeros((2,3,n_plies)) # 2 for bottom and top values; 3 for 1, 2, 12 strains; per ply
    z_midplane = t * n_plies / 2
    for k in range(n_plies):
        if active_plies[k]==True:
            T_sigma, T_epsilon = transformation_matrices(angles[k])
            T_epsilon = np.block([
                [T_epsilon,np.zeros((3,3))],
                [np.zeros((3,3)), T_epsilon]
            ])

            strain_12_mid = T_epsilon @ strain_xy_mid
            z_k_bottom = t * k - z_midplane
            z_k_top = t * (k+1) - z_midplane

            #Bottom Strain
            strain[0, :, k] = [strain_12_mid[0] + z_k_bottom * strain_12_mid[3],
                               strain_12_mid[1] + z_k_bottom * strain_12_mid[4],
                               strain_12_mid[2] + z_k_bottom * strain_12_mid[5]]

            #Top Strain
            strain[1, :, k] = [strain_12_mid[0] + z_k_top * strain_12_mid[3],
                               strain_12_mid[1] + z_k_top * strain_12_mid[4],
                               strain_12_mid[2] + z_k_top * strain_12_mid[5]]

    return strain

def lamina_stress12(strain, Q_0, Q_0degraded, n_plies, failed_once):
    # calculates stresses for each ply in local ply coordinates
    # Inputs: 2 x 3 x n_plies array with 1, 2, and 12 strain values for top and bottom of each ply
    #         Q_0 array per initial ply at 0 deg
    #         Q_0degraded per degraded ply at 0 deg
    #         n_plies number of plies
    #         true/false array stating whether a ply is able to carry loads (by default all true)
    # Outputs: 2x3xn_plies array with 1, 2, and 12 stress values for top and bottom of each ply

    stress = np.zeros((2,3,n_plies))
    for i in range(n_plies):
        if failed_once[i]==False:
            stress[0,:,i] = Q_0 @ strain[0,:,i]
            stress[1, :, i] = Q_0 @ strain[1,:,i]

        else:
            stress[0, :, i] = Q_0degraded @ strain[0, :, i]
            stress[1, :, i] = Q_0degraded @ strain[1, :, i]

    return stress

def failure(stress, strain, E1, Ef, v12f, Xt, Xc, Yt, Yc, Stc, n_plies, active_plies1, failed_once1):
    # Puck failure criterion
    # Inputs: 2x3xn_plies array with 1, 2, and 12 stress values for top and bottom of each ply (in MPa);
    #         2x3xn_plies array with 1, 2, and 12 strain values for top and bottom of each ply (unitless strain);
    #         material properties with units:
    #           E1, Ef, Xt, Xc, Yt, Yc, Stc - MPa
    #           v12f - unitless
    #         number of plies;
    #         true/false array stating whether a ply is able to carry loads (by default all true);
    #         true/false array stating whether a ply failed in transverse direction only
    #                            (0.15 degradation rule is applied) ((by default all false))
    #
    # Output: updated true/false array stating whether a ply is able to carry loads;
    #         updated true/false array stating whether a ply failed in transverse direction only
    #         (0.15 degradation rule is applied)

    # Puck failure parameters
    p_parallel_t = 0.35  # [-]
    p_parallel_c = 0.30  # [-]
    p_perp_t = 0.25  # [-]
    p_perp_c = 0.25  # [-]
    Ytdegraded = 0.15 * Yt  # MPa
    Ycdegraded = 0.15 * Yc  # MPa
    Stz = np.sqrt(((1 + Yt / Yc) / (3 + 5 * Yt / Yc)) * Yt * Yc)  # MPa

    active_plies = active_plies1.copy()
    failed_once = failed_once1.copy()

    for i in range(n_plies):
        if active_plies[i] == True:
            a=0
            if failed_once1[i] == True:
                Yt = Ytdegraded
                Yc = Ycdegraded
            for j in range(2):
                # Fibre tension or compression failure
                strainL = strain[j,0,i]
                stressT = stress[j,1,i]
                stressTL = stress[j,2,i]
                eps = 1e-12
                stressTL = max(abs(stressTL), eps)
                if stressT >= 0:
                    snnL = p_parallel_t
                    snnT = p_perp_t
                else:
                    snnL = p_parallel_c
                    snnT = p_perp_c
                R = Yc / (2 * (1 + snnT))
                Sc = Stc * np.sqrt(1 + 2 * snnT)
                bracket = (strainL + v12f / Ef * 1.1 * stressT)
                if bracket >= 0:
                    X = Xt
                else:
                    X = Xc
                fibre = E1/X * np.abs(bracket)

                if fibre >= 1:
                    active_plies[i] = False
                    failed_once[i] = True
                    continue

                # Matrix tension failure (stressT>0)
                if stressT >= 0:
                    matrix_tension = np.sqrt(
                        (stressTL / Stc) ** 2 + (1 - snnL * Yt / Stc) ** 2 * (stressT / Yt) ** 2) + snnL * stressT / Stz
                    if matrix_tension >= 1:
                        if failed_once[i] == True:
                            active_plies[i] = False
                            continue
                        else:
                            a += 1

                elif np.abs(stressT/stressTL) <= R/Sc:
                    matrix_comp1 = np.sqrt((stressTL / Stc) ** 2 + (snnL * Yt / Stc) ** 2) + snnL * stressT / Stc
                    if matrix_comp1 >= 1:
                        if failed_once[i] == True:
                            active_plies[i] = False
                            continue
                        else:
                            a += 1

                elif np.abs(stressTL/stressT) <= Sc/R:
                    matrix_comp2 = -Yc / stressT * ((stressTL / Yc * R / Stc) ** 2 + (stressT / Yc) ** 2)
                    if matrix_comp2 >= 1:
                        if failed_once[i] == True:
                            active_plies[i] = False
                            continue
                        else:
                            a += 1

                if j == 1:
                    if a>0:
                        failed_once[i] = True

    return active_plies, failed_once


