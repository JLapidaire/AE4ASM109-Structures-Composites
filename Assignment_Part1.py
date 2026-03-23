import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from progressive_damage import failure,lamina_strains_inplane,lamina_stress_inplane,calc_A,calc_Q_laminate,inplane_matrices

thetas = np.linspace(0,90,num=180)
phis = thetas


def EngineeringConstants(A,h):
    ### Engineering Constants of laminate ####
    # Inputs:   A-Matrix (of the ABD Matrix), 3x3
    #           h (total laminate thickness)
    Axx = A[0,0]
    Ayy = A[1,1]
    Axy = A[0,1]
    Ass = A[2,2]

    Ex = (Axx * Ayy - Axy**2) / (h * Ayy)
    Ey = (Axx * Ayy - Axy**2) / (h * Axx)
    vxy = Axy / Ayy
    vyx = Axy / Axx
    Gxy = Ass / h

    return Ex, Ey, vxy, vyx, Gxy

def LaminateConstants(thetas,phis,plotting):
    E1 = 172.3 * 10 ** 3  # MPa
    E2 = 10.2 * 10 ** 3  # MPa
    G12 = 5.58 * 10 ** 3  # MPa
    v12 = 0.25  # [-]
    t = 0.125 #mm

    Ex_data = np.zeros((len(thetas),len(phis)))
    Ey_data = np.zeros((len(thetas),len(phis)))
    vxy_data = np.zeros((len(thetas),len(phis)))
    vyx_data = np.zeros((len(thetas),len(phis)))
    Gxy_data = np.zeros((len(thetas),len(phis)))

    for theta in thetas:
        for phi in phis:
            angles = np.array([+theta,-theta,phi,phi,phi,phi,phi,phi,-theta,+theta,+theta,-theta,phi,phi,phi,phi,phi,phi,-theta,+theta])
            angles = angles * (np.pi / 180)
            n_plies = len(angles)

            S, Q_0 = inplane_matrices(E1, E2, v12, G12)

            Q_0degraded = Q_0.copy()
            Q_0degraded[1, 1] *= 0.15
            Q_0degraded[0, 1] *= 0.15
            Q_0degraded[1, 0] *= 0.15
            Q_0degraded[2, 2] *= 0.15

            active_plies = np.ones(n_plies, dtype=bool)
            failed_once = np.zeros(n_plies, dtype=bool)

            Q_all = calc_Q_laminate(Q_0, Q_0degraded, angles, active_plies, failed_once, n_plies)
            ABD = calc_A(Q_all, t, n_plies)

            Ex, Ey, vxy, vyx, Gxy = EngineeringConstants(ABD,t*n_plies)
            Ex_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += (Ex * 10**-3)
            Ey_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += (Ey * 10**-3)
            vxy_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += vxy
            vyx_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += vyx
            Gxy_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += (Gxy * 10**-3)
            print('for phi = ' + str(phi) + ',for theta = ' + str(theta) + str([Ex, Ey, vxy, vyx, Gxy]))

    if plotting:
        data = np.array([Ex_data,Ey_data,vxy_data,vyx_data,Gxy_data])
        titles = np.array(['Ex (GPa)','Ey (GPa)','vxy (-)','vyx (-)','Gxy (GPa)'])
        thetas2,phis2 = np.meshgrid(thetas,phis)
        for I in range(5):
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(thetas2,phis2,data[I],cmap='viridis')
            ax.set_xlabel('theta (deg)')
            ax.set_ylabel('phi (deg)')
            ax.set_zlabel(titles[I])
            plt.show()

    return Ex_data,Ey_data,vxy_data,vyx_data,Gxy_data

Ex_data,Ey_data,vxy_data,vyx_data,Gxy_data = LaminateConstants(thetas,phis,1)


