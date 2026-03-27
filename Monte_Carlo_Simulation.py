import numpy as np
import matplotlib.pyplot as plt
from progressive_damage import failure,lamina_strains_inplane,lamina_stress_inplane,calc_A,calc_Q_laminate,inplane_matrices

def RandomVariable(Mean,STD):
    return np.random.normal(loc=Mean,scale=STD)

def GenerateProperties():
    E1 = RandomVariable(172.3,9.28) * 10**3 #MPa
    E2 = RandomVariable(10.2,3.58) * 10**3 #MPa
    v12 = RandomVariable(0.25,0.018) #-
    G12 = RandomVariable(5.58,1.97) * 10**3 #MPa
    Xt = RandomVariable(1923,188.3) #MPa
    Yt = RandomVariable(84,8.2) #MPa
    Xc = 1480 #MPa
    Yc = 220 #MPa (Not Distributed?)
    S = RandomVariable(144.5,7.33) #MPa

    Properties = np.array([E1,E2,v12,G12,Xt,Yt,Xc,Yc,S])

    return Properties

def n_basedSimulation(plotting,load):
    # Input Parameters From Assignment
    Plies = [0,90,45,-45,-45,45,90,0,0,90,45,-45,-45,45,90,0]
    N = load #N/mm at 45 degrees
    Loading = np.array([np.cos(np.radians(45))*N,np.sin(np.radians(45))*N,0]) #Force Vector 1x3

    #Convergence Settings
    min_failures = 10
    target_error = 1e-4
    R_error = 1

    Pfs = np.array([])
    failure_total = 0
    n = 1
    while R_error > target_error or failure_total < min_failures:
        failure = FirstPlyFailure(Plies,Loading)
        failure_total += failure
        Pf = failure_total / n
        Pfs = np.append(Pfs,Pf)

        #Standard Error
        S_error = np.std(Pfs) / np.sqrt(n)
        #Relative Error
        R_error = S_error / np.average(Pfs)

        if failure:
            print('Failure detected at n = ' + str(n))
            print('Current Pf = ' + str(Pf) + ' and relative error of ' + str(R_error))

        n += 1

    if plotting:
        plt.figure(figsize=(16,9)) #16:9 Aspect Ratio
        plt.plot(np.arange(1,n),Pfs*100)
        plt.title('Converge Plot for N = ' + str(N) + ' N/mm')
        plt.ylabel('Average Failure Probability [%]')
        plt.xlabel('Number of Simulations [-]')
        plt.grid()
        plt.show()

    return Pfs*100

def FirstPlyFailure(ply_angles,loading):
    ply_angles = np.radians(ply_angles) #Convert Plies to Radians
    n_plies = len(ply_angles) 

    Properties = GenerateProperties() #Generate Random Ply Properties
    E1 = Properties[0]
    E2 = Properties[1]
    v12 = Properties[2]
    G12 = Properties[3]
    Xt = Properties[4]
    Yt = Properties[5]
    Xc = Properties[6]
    Yc = Properties[7]
    Stc = Properties[8]

    #Not-Given Properties
    Ef = 230 * 10**3 #MPa
    v12f = 0.25
    t = 0.125 #mm

    S, Q_0 = inplane_matrices(E1, E2, v12, G12)

    Q_0degraded = Q_0.copy()
    Q_0degraded[1, 1] *= 0.15
    Q_0degraded[0, 1] *= 0.15
    Q_0degraded[1, 0] *= 0.15
    Q_0degraded[2, 2] *= 0.15

    # reset laminate state for this loading ratio
    active_plies = np.ones(n_plies, dtype=bool)
    failed_once = np.zeros(n_plies, dtype=bool)

    Q_all = calc_Q_laminate(Q_0, Q_0degraded, ply_angles, active_plies, failed_once, n_plies)
    ABD = calc_A(Q_all, t, n_plies)
    abd = np.linalg.inv(ABD)

    fpf_recorded = False

    # save old state explicitly
    active_old = active_plies.copy()
    failed_old = failed_once.copy()

    strain = lamina_strains_inplane(ply_angles, abd, loading, n_plies, active_plies)
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
            fpf_recorded = True

    return fpf_recorded

load = 783 #N/mm
Pfs = n_basedSimulation(1,load)
print('Converged Failure Probability at Pf = ' + str(Pfs[-1]) + ', at n = ' + str(len(Pfs)))
