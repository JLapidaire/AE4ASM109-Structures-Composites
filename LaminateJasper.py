import numpy as np
import scipy.linalg as la

class Lamina:
    def __init__(self,properties):
        self.E1 = properties[0] #Tensile Modulus in fibre direction
        self.E2 = properties[1] #Tensile Modulus in transverse direction
        self.G12 = properties[2] #Shear Modulus
        self.v12 = properties[3] #Major Poisson Ratio
    
    def QMatrix(self,theta):
        S11 = 1/self.E1
        S22 = 1/self.E2
        S12 = -self.v12/self.E1
        S66 = 1/self.G12
        D = S11 * S22 - S12**2

        Q11 =  S22 * (1/D)
        Q22 =  S11 * (1/D)
        Q12 = -S12 * (1/D)
        Q66 = 1 / S66

        n = np.sin(np.radians(theta))
        m = np.cos(np.radians(theta))

        Q11t = (m**4)*Q11 + (n**4)*Q22 + 2*(m**2)*(n**2)*Q12 + 4*(m**2)*(n**2)*Q66
        Q22t = (n**4)*Q11 + (m**4)*Q22 + 2*(m**2)*(n**2)*Q12 + 4*(m**2)*(n**2)*Q66
        Q12t = (m**2)*(n**2)*Q11 + (m**2)*(n**2)*Q22 + (m**4 + n**4)*Q12 - 4*(m**2)*(n**2)*Q66
        Q66t = (m**2)*(n**2)*Q11 + (m**2)*(n**2)*Q22 -2*(m**2)*(n**2)*Q12 + ((m**2 - n**2)**2)*Q66
        Q16t = (m**3)*n*Q11 - m*(n**3)*Q22 + (m*(n**3)-(m**3)*n)*Q12 + 2*(m*(n**3)-(m**3)*n)*Q66
        Q26t = m*(n**3)*Q11 - (m**3)*n*Q22 + ((m**3)*n-m*(n**3))*Q12 + 2*((m**3)*n-m*(n**3))*Q66

        self.Q_rot = np.array([[Q11t,Q12t,Q16t],
                               [Q12t,Q22t,Q26t],
                               [Q16t,Q26t,Q66t]])
        return self.Q_rot

class Laminate:
    def __init__(self,plies,thicknesses,properties):
        self.pList = plies #Array of each ply orientation, bottom to top
        self.tList = thicknesses #Array of each ply thickness, same format as ply orientations
        self.propList = properties #Array of properties of each ply, same format

        self.t_tot = np.sum(self.tList) #Laminate thickness
        t_mid = self.t_tot / 2 #Midline Point
        self.t_coords =  np.insert(np.cumsum(self.tList),0,0) - t_mid #Laminate coordinates

    def ABDMatrix(self):
        self.ABD = np.zeros((6,6))

        for i in range(len(self.pList)):
            Lam = Lamina(properties=self.propList[i])
            Q = Lam.QMatrix(theta=self.pList[i])
            self.ABD[0:3,0:3] += (Q   * (self.t_coords[i+1]-self.t_coords[i]))
            self.ABD[3:6,0:3] += (Q/2 * (self.t_coords[i+1]**2 -self.t_coords[i]**2))
            self.ABD[0:3,3:6] += (Q/2 * (self.t_coords[i+1]**2 -self.t_coords[i]**2))
            self.ABD[3:6,3:6] += (Q/3 * (self.t_coords[i+1]**3 -self.t_coords[i]**3))

        return self.ABD

    def InverseABD(self):
        self.InvABD = np.zeros((6,6))

        A = self.ABD[0:3,0:3]
        B = self.ABD[3:6,0:3]
        D = self.ABD[3:6,3:6]

        self.InvABD[0:3,0:3] += la.inv(A) + la.inv(A) @ B @ la.inv(D - B @ la.inv(A) @ B) @ B @ la.inv(A)
        self.InvABD[3:6,0:3] += - A @ B @ la.inv(D - B @ la.inv(A) @ B)
        self.InvABD[0:3,3:6] += - A @ B @ la.inv(D - B @ la.inv(A) @ B)
        self.InvABD[3:6,3:6] += la.inv(D - B @ la.inv(A) @ B)

        return self.InvABD

plies = [0,45,-45,90]
thickness = [0.125,0.125,0.125,0.125]
properties = np.tile([140000,10000,5000,0.3],(len(plies),1))

Lamin = Laminate(plies=plies,thicknesses=thickness,properties=properties)
ABD = np.round(Lamin.ABDMatrix(),2)
InvABD = np.round(Lamin.InverseABD(),2)
print(la.inv(ABD))
print(InvABD)
    