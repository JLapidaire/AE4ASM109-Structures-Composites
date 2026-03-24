import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d
import progressive_damage as pd
import ABD as abd

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

    S, Q_0 = pd.inplane_matrices(E1, E2, v12, G12)

    Q_0degraded = Q_0.copy()
    Q_0degraded[1, 1] *= 0.15
    Q_0degraded[0, 1] *= 0.15
    Q_0degraded[1, 0] *= 0.15
    Q_0degraded[2, 2] *= 0.15

    for theta in thetas:
        for phi in phis:
            angles = np.array([+theta,-theta,phi,phi,phi,phi,phi,phi,-theta,+theta,+theta,-theta,phi,phi,phi,phi,phi,phi,-theta,+theta])
            angles = angles * (np.pi / 180)
            n_plies = len(angles)

            active_plies = np.ones(n_plies, dtype=bool)
            failed_once = np.zeros(n_plies, dtype=bool)

            Q_all = pd.calc_Q_laminate(Q_0, Q_0degraded, angles, active_plies, failed_once, n_plies)
            ABD = pd.calc_A(Q_all, t, n_plies)

            Ex, Ey, vxy, vyx, Gxy = EngineeringConstants(ABD,t*n_plies)
            Ex_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += (Ex * 10**-3)
            Ey_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += (Ey * 10**-3)
            vxy_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += vxy
            vyx_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += vyx
            Gxy_data[np.argwhere(thetas==theta),np.argwhere(phis==phi)] += (Gxy * 10**-3)
            #print('for phi = ' + str(phi) + ',for theta = ' + str(theta) + str([Ex, Ey, vxy, vyx, Gxy]))

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

def LaminateLoading(plotting):
    #Material Properties
    E1 = 172.3 * 10 ** 3  # MPa
    E2 = 10.2 * 10 ** 3  # MPa
    G12 = 5.58 * 10 ** 3  # MPa
    v12 = 0.25  # [-]
    t = 0.125 #mm

    #Loading Properties
    Nx = 300 * 10**-3 #N/mm
    Ns = 25 * 10**-3 #N/mm
    My = 18 * 10**3 #N
    F = np.array([Nx,0,Ns,0,My,0])

    angles = np.array([0,45,-45,90,-60,30,0])
    angles = angles * (np.pi / 180)
    n_plies = len(angles)

    S, Q_0 = pd.inplane_matrices(E1, E2, v12, G12)

    Q_0degraded = Q_0.copy()
    Q_0degraded[1, 1] *= 0.15
    Q_0degraded[0, 1] *= 0.15
    Q_0degraded[1, 0] *= 0.15
    Q_0degraded[2, 2] *= 0.15

    active_plies = np.ones(n_plies, dtype=bool)
    failed_once = np.zeros(n_plies, dtype=bool)

    Q_all = abd.calc_Q_laminate(Q_0, Q_0degraded, angles, active_plies, failed_once, n_plies)
    ABD = abd.calc_ABD(Q_all, t, n_plies)

    strain = abd.lamina_strains12(angles,ABD,F,t,n_plies,active_plies)
    stress = abd.lamina_stress12(strain,Q_0,Q_0degraded,n_plies,failed_once)

    if plotting:
        strain_bottom = strain[0]
        strain_top = strain[1]
        stress_bottom = stress[0]
        stress_top = stress[1]

        t_bottom = 0 + np.arange(0,n_plies) * t
        t_top = t + np.arange(0,n_plies) * t
        t_stitch = StitchFunctions(t_bottom,t_top)

        # Stitch Strains Together For Plotting
        exx_stitch = StitchFunctions(strain_bottom[0],strain_top[0])
        eyy_stitch = StitchFunctions(strain_bottom[1],strain_top[1])
        exy_stitch = StitchFunctions(strain_bottom[2],strain_top[2])

        # Stitch Stresses Together For Plotting
        Sxx_stitch = StitchFunctions(stress_bottom[0],stress_top[0])
        Syy_stitch = StitchFunctions(stress_bottom[1],stress_top[1])
        Sxy_stitch = StitchFunctions(stress_bottom[2],stress_top[2])

        fig,ax = plt.subplots(2,3)
        #Plot Strains
        ax[0,0].plot(exx_stitch,t_stitch)
        ax[0,1].plot(eyy_stitch,t_stitch)
        ax[0,2].plot(exy_stitch,t_stitch)
        #Plot Stresses
        ax[1,0].plot(Sxx_stitch,t_stitch)
        ax[1,1].plot(Syy_stitch,t_stitch)
        ax[1,2].plot(Sxy_stitch,t_stitch)

        titles = ['Longitudinal Strain','Transverse Strain','Shear Strain','Longitudinal Stress','Transverse Stress','Shear Stress']
        xlabels = ['Strain [-]','Strain [-]','Strain [-]','Stress [-]','Stress [-]','Stress [-]']
        ylabels = ['Thickness [mm]','','','Thickness [mm]','','']
        index = 0

        for axis in ax.flat:
            axis.set(title=titles[index],xlabel=xlabels[index],ylabel=ylabels[index])
            axis.yaxis.set_major_locator(ticker.MultipleLocator(t))
            axis.grid()
            index += 1

        plt.show()

    return strain, stress

def StitchFunctions(a,b):
    'Stitches together to arrays element-per-element'
    stitch = np.empty((a.size + b.size,), dtype=a.dtype)
    stitch[0::2] = a
    stitch[1::2] = b

    return stitch

Ex_data,Ey_data,vxy_data,vyx_data,Gxy_data = LaminateConstants(thetas,phis,1)
strain, stress = LaminateLoading(1)



