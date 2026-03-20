import numpy as np
import matplotlib.pyplot as plt


#######################
#   strain = S stress
#   stress = C strain

### UD LAMINA ###

### IN PLANE
def inplane_matrices(E1, E2, v12, G12):
    # Finds in S and Q matrices for on-axis lamina
    v21 = E2 / E1 * v12

    S = np.array([[1 / E1, -v21 / E2, 0],
                  [-v12 / E1, 1 / E2, 0],
                  [0, 0, 1 / G12]])

    Q = np.linalg.inv(S)
    return S, Q


### ROTATION MATRICES
def transformation_matrices(theta):
    # Finds transformation for off-axis lamina
    m = np.cos(theta)
    n = np.sin(theta)
    T_sigma = np.array([[m ** 2, n ** 2, 2 * m * n],
                        [n ** 2, m ** 2, -2 * m * n],
                        [-m * n, m * n, (m ** 2 - n ** 2)]])

    T_epsilon = np.array([[m ** 2, n ** 2, m * n],
                          [n ** 2, m ** 2, -m * n],
                          [-2 * m * n, 2 * m * n, (m ** 2 - n ** 2)]])
    return T_sigma, T_epsilon


def offaxis_matrices(Q, theta):
    T_sigma, T_epsilon = transformation_matrices(theta)
    # S_theta = np.linalg.inv(T_epsilon) @ S @ T_sigma
    Q_theta = np.linalg.inv(T_sigma) @ Q @ T_epsilon
    return Q_theta


def calc_A(Q_all, t, n_plies):
    # Inputs:   Array consisting of Q for each ply in fibre orientations,
    #           thickness of each ply (assumes the same for all),
    #           number of plies
    # Output:   A matrix, can be used instead of full ABD if only in plane loads are applied, reduces computation time
    A = np.zeros((3, 3))
    T_midplane = t * n_plies / 2  # define where midplane is
    for k in range(n_plies):
        z_k = t * k - T_midplane
        z_k_next = t * (k + 1) - T_midplane
        A[:, :] += Q_all[:, :, k] * (z_k_next - z_k)
    return A


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
    Q_all = np.zeros((3, 3, n_plies))
    for i in range(n_plies):
        # this part checks if the given ply is still active (true) or has failed (false)
        # and if it is active, was it the first or second failure
        if active_plies[i] == True:
            if failed_once[i] == True:
                Q_angle = offaxis_matrices(Q_0degraded, angles[i])
                Q_all[:, :, i] = Q_angle
            else:
                Q_angle = offaxis_matrices(Q_0, angles[i])
                Q_all[:, :, i] = Q_angle
        else:
            Q_all[:, :, i] = np.zeros((3, 3))
    return Q_all


def lamina_strains_inplane(angles, abd, F, n_plies, active_plies):
    # converts the strain in 12 direction in midplane to strains in each ply IN LOCAL (PLY) COORDINATE SYSTEMS.
    # returns 3 x number of plies matrix of strains for all plies in local ply coordinates
    # Inputs: array with orientation angles of each ply
    #         ABD matrix
    #         6x1 force vector F
    #         thickness of the plies
    #         number of plies

    strain_xy_mid = abd @ F
    strain = np.zeros((1, 3, n_plies))  # 2 for bottom and top values; 3 for x, y, shear; per ply
    # z_midplane = t * n_plies / 2
    for k in range(n_plies):
        if active_plies[k] == True:
            T_sigma, T_epsilon = transformation_matrices(angles[k])

            strain_12_mid = T_epsilon @ strain_xy_mid
            # z_k_bottom = t * k - z_midplane
            # z_k_top = t * (k + 1) - z_midplane

            strain[0, :, k] = [strain_12_mid[0],
                               strain_12_mid[1],
                               strain_12_mid[2]]
    return strain


def lamina_stress_inplane(strain, Q_0, Q_0degraded, n_plies, failed_once):
    # calculates stresses for each ply in local ply coordinates
    # Inputs: Q_0 array per initial ply at 0 deg
    #         Q_0degraded per degraded ply at 0 deg
    stress = np.zeros((1, 3, n_plies))
    for i in range(n_plies):
        if failed_once[i] == False:
            stress[0, :, i] = Q_0 @ strain[0, :, i]

        else:
            stress[0, :, i] = Q_0degraded @ strain[0, :, i]

    return stress


def failure(stress, strain, E1, Ef, v12f, Xt, Xc, Yt, Yc, Stc, n_plies, active_plies1, failed_once1):
    ################# FIND Ef AND v12f !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ##### find source for why snnL = 0.25 and snnT = 0.25
    snnL = 0.25
    snnT = 0.25
    Stz = 144.5  # placeholder
    R = Yc / (2 * (1 + snnT))
    Sc = Stc * np.sqrt(1 + 2 * snnT)
    active_plies = active_plies1.copy()
    failed_once = failed_once1.copy()
    for i in range(n_plies):
        if active_plies[i] == True:
            a = 0

            # Fibre tension or compression failure
            strainL = strain[0, 0, i]
            stressT = stress[0, 1, i]
            stressTL = stress[0, 2, i]
            eps = 1e-12
            stressTL = max(abs(stressTL), eps)
            bracket = (strainL + v12f / Ef * 1.1 * stressT)
            if bracket >= 0:
                X = Xt
            else:
                X = Xc

            fibre = E1 / X * np.abs(bracket)

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

            elif np.abs(stressT / stressTL) <= R / Sc:
                matrix_comp1 = np.sqrt((stressTL / Stc) ** 2 + (snnL * Yt / Stc) ** 2) + snnL * stressT / Stc
                if matrix_comp1 >= 1:
                    if failed_once[i] == True:
                        active_plies[i] = False
                        continue
                    else:
                        a += 1

            elif np.abs(stressTL / stressT) <= Sc / R:
                matrix_comp2 = -Yc / stressT * ((stressTL / Yc * R / Stc) ** 2 + (stressT / Yc) ** 2)
                if matrix_comp2 >= 1:
                    if failed_once[i] == True:
                        active_plies[i] = False
                        continue
                    else:
                        a += 1


            if a > 0:
                failed_once[i] = True

    return active_plies, failed_once

if __name__ == '__main__': #Only Run this code if this file is run (and not if it is being imported)
    # material data
    # UD lamina
    E1 = 172.3 * 10 ** 3  # MPa
    E2 = 10.2 * 10 ** 3  # MPa
    G12 = 5.58 * 10 ** 3  # MPa
    v12 = 0.25  # [-]
    '''thickness is still a placeholder value'''
    t = 0.125  # mm
    # Failure criteria
    Xt = 1923  # MPa
    Xc = 1480  # MPa
    Yt = 84  # MPa
    Yc = 220  # MPa
    Stc = 144.5  # MPa
    Ef = E1
    v12f = v12

    base = [0, 45, -45, 90, 30]
    sym1 = (base) + (base)[::-1]
    sym2 = sym1 + sym1[::-1]
    sym3 = sym2 + sym2[::-1]
    angles = np.pi / 180 * np.array(base)
    n_plies = int(len(base))  # total number of plies



    S, Q_0 = inplane_matrices(E1, E2, v12, G12)

    Q_0degraded = Q_0.copy()
    Q_0degraded[1, 1] *= 0.15
    Q_0degraded[0, 1] *= 0.15
    Q_0degraded[1, 0] *= 0.15
    Q_0degraded[2, 2] *= 0.15



    number = 90
    ratios = np.zeros((2, number))

    for i in range(number):
        angle = i * 2 * np.pi / number
        ratios[0, i] = np.cos(angle)
        ratios[1, i] = np.sin(angle)

    first_ply_failure_NxNy = np.full((2, number), np.nan)
    last_ply_failure_NxNy = np.full((2, number), np.nan)
    first_ply_failure_NyNs = np.full((2, number), np.nan)
    last_ply_failure_NyNs = np.full((2, number), np.nan)

    # load increment magnitude
    load_step = 1.0
    max_steps = 1000

    for i in range(number):
        # reset laminate state for this loading ratio
        active_plies = np.ones(n_plies, dtype=bool)
        failed_once = np.zeros(n_plies, dtype=bool)

        Q_all = calc_Q_laminate(Q_0, Q_0degraded, angles, active_plies, failed_once, n_plies)
        ABD = calc_A(Q_all, t, n_plies)
        abd = np.linalg.inv(ABD)

        # loading direction for this ratio
        direction = np.array([ratios[0, i], ratios[1, i], 0.0])

        Fxy = load_step * direction
        Fxy_increment = load_step * direction

        fpf_recorded = False
        step_counter = 0

        while np.any(active_plies):
            step_counter += 1
            if step_counter > max_steps:
                print(f"max_steps reached at ratio {i}")
                break

            # save old state explicitly
            active_old = active_plies.copy()
            failed_old = failed_once.copy()

            strain = lamina_strains_inplane(angles, abd, Fxy, n_plies, active_plies)
            stress = lamina_stress_inplane(strain, Q_0, Q_0degraded, n_plies, failed_once)

            active_new, failed_new = failure(
                stress, strain, E1, Ef, v12f, Xt, Xc, Yt, Yc, Stc,
                n_plies, active_plies, failed_once
            )

            state_changed = (
                    (not np.array_equal(active_new, active_old)) or
                    (not np.array_equal(failed_new, failed_old))
            )

            if state_changed:
                # first state change = first ply failure
                if not fpf_recorded:
                    first_ply_failure_NxNy[0, i] = Fxy[0]
                    first_ply_failure_NxNy[1, i] = Fxy[1]
                    fpf_recorded = True

                # update state
                active_plies = active_new
                failed_once = failed_new

                # if all plies are gone, record LPF at this same load
                if not np.any(active_plies):
                    last_ply_failure_NxNy[0, i] = Fxy[0]
                    last_ply_failure_NxNy[1, i] = Fxy[1]
                    break

                # rebuild stiffness and re-analyze at SAME load
                Q_all = calc_Q_laminate(Q_0, Q_0degraded, angles, active_plies, failed_once, n_plies)
                ABD = calc_A(Q_all, t, n_plies)
                abd = np.linalg.inv(ABD)

                continue

            # no damage at this load -> increase load
            Fxy = Fxy + Fxy_increment

    first_ply_failure_NxNy *= 2**3
    last_ply_failure_NxNy *= 2**3

    print("FPF NxNy:")
    print(first_ply_failure_NxNy)

    print("LPF NxNy:")
    print(last_ply_failure_NxNy)

    plt.figure(figsize=(7, 7))
    plt.scatter(first_ply_failure_NxNy[0, :], first_ply_failure_NxNy[1, :], label='FPF')
    plt.scatter(last_ply_failure_NxNy[0, :], last_ply_failure_NxNy[1, :], label='LPF')
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel('Nx')
    plt.ylabel('Ny')
    plt.title('Failure Envelope (Nx-Ny)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
