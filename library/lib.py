import symengine as sm
import sympy as sp
import numpy as np
from numpy import sqrt, exp, sin, cos, sinh, cosh, pi
from numpy import arccos as acos
from numpy import arcsin as asin
from numpy import arccosh as acosh
from numpy import arcsinh as asinh
from numpy import arctan2 as atan2
import matplotlib.pyplot as plt



x, y, z = sm.symbols('x y z', real=True, positive=True)
r, phi, theta = sm.symbols('r phi theta', real=True, positive=True)

x_pol = r*sm.sin(theta)*sm.cos(phi)
y_pol = r*sm.sin(theta)*sm.sin(phi)
z_pol = r*sm.cos(theta)

r_cart = sm.sqrt(x**2+y**2+z**2)
theta_cart = sm.acos(z/r_cart)
phi_cart = sm.atan2(y,x)

r_cart_2d = sm.sqrt(x**2+y**2)
phi_cart_2d = sm.atan2(y,x)





class Metric_polar():

    def __init__(self, g11, g22=None, g33=None, g12=None, g13=None, g21=None, g23=None, g31=None, g32=None):

        if g11 is not None:
            self.g11 = g11
        else:
            self.g11 = '0'
        if g12 is not None:
            self.g12 = g12
        else:
            self.g12 = '0'
        if g13 is not None:
            self.g13 = g13
        else:
            self.g13 = '0'

        if g21 is not None:
            self.g21 = g21
        else:
            self.g21 = '0'
        if g22 is not None:
            self.g22 = g22
        else:
            self.g12 = '0'
        if g23 is not None:
            self.g23 = g23
        else:
            self.g23 = '0'

        if g31 is not None:
            self.g31 = g31
        else:
            self.g31 = '0'
        if g32 is not None:
            self.g32 = g32
        else:
            self.g32 = '0'
        if g33 is not None:
            self.g33 = g33
        else:
            self.g33 = '0'



    def inverse(self):

        g_ij = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in g_ij[:,0]) and all(val == 0 for val in g_ij[0,:]):
            g_ij.col_del(0)
            g_ij.row_del(0)
        if all(val == 0 for val in g_ij[:,2]) and all(val == 0 for val in g_ij[2,:]):
            g_ij.col_del(2)
            g_ij.row_del(2)
        
        gij = g_ij.inv()

        gij_object = Metric_polar(g11=gij[0,0], g22=gij[1,1], g33=gij[2,2],
                                  g12=gij[0,1], g13=gij[0,2], g21=gij[1,0],
                                  g23=gij[1,2], g31=gij[2,0], g32=gij[2,1])
        
        return gij_object
    


    def print(self, comp=None):

        g_ij = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in g_ij[:,0]) and all(val == 0 for val in g_ij[0,:]):
            g_ij = g_ij.col_del(0)
            g_ij = g_ij.row_del(0)
        if all(val == 0 for val in g_ij[:,2]) and all(val == 0 for val in g_ij[2,:]):
            g_ij = g_ij.col_del(2)
            g_ij = g_ij.row_del(2)

        if comp is None:
            print(g_ij)
        else:
            print(g_ij[comp[0], comp[1]])



    def pol_to_cart(self):

        g_polar = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in g_polar[:,0]) and all(val == 0 for val in g_polar[0,:]):
            g_polar.col_del(0)
            g_polar.row_del(0)
        if all(val == 0 for val in g_polar[:,2]) and all(val == 0 for val in g_polar[2,:]):
            g_polar.col_del(2)
            g_polar.row_del(2)
        
        vector1 = sm.Matrix([[r_cart], [theta_cart], [phi_cart]])
        vector2 = sm.Matrix([[x],[y],[z]])
        vector1_2d = sm.Matrix([[r_cart_2d], [phi_cart_2d]])
        vector2_2d = sm.Matrix([[x],[y]])

        if np.shape(g_polar)==(3, 3):
            J_pol_to_cart = vector1.jacobian(vector2)
            g_polar_xyz = g_polar.subs(r, r_cart)
            g_polar_xyz = g_polar_xyz.subs(theta, theta_cart)
            g_polar_xyz = g_polar_xyz.subs(phi, phi_cart)
        if np.shape(g_polar)==(2, 2):
            J_pol_to_cart = vector1_2d.jacobian(vector2_2d)
            g_polar_xyz = g_polar.subs(r, r_cart_2d)
            g_polar_xyz = g_polar_xyz.subs(theta, sm.symbols('pi')/2)
            g_polar_xyz = g_polar_xyz.subs(phi, phi_cart_2d)
        

        

        g_cart = J_pol_to_cart.transpose() * g_polar_xyz * J_pol_to_cart
        g_cart = g_cart.expand()

        g_cart_object = Metric_cartesian(g11=g_cart[0,0], g22=g_cart[1,1], g33=g_cart[2,2],
                                         g12=g_cart[0,1], g13=g_cart[0,2], g21=g_cart[1,0],
                                         g23=g_cart[1,2], g31=g_cart[2,0], g32=g_cart[2,1])
        
        return g_cart_object



    def to_permittivity(self):

        gij = sm.Matrix([[self.g11, self.g12, self.g13],
                         [self.g21, self.g22, self.g23],
                         [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in gij[:,0]) and all(val == 0 for val in gij[0,:]):
            gij.col_del(0)
            gij.row_del(0)
        if all(val == 0 for val in gij[:,2]) and all(val == 0 for val in gij[2,:]):
            gij.col_del(2)
            gij.row_del(2)
        
        gij = gij.subs(r, r_cart)
        gij = gij.subs(theta, theta_cart)
        gij = gij.subs(phi, phi_cart)

        
        det_sqrt = sm.sqrt(gij.det())

        E = det_sqrt * gij

        E_object = Permittivity(g11=E[0,0], g22=E[1,1], g33=E[2,2],
                                g12=E[0,1], g13=E[0,2], g21=E[1,0],
                                g23=E[1,2], g31=E[2,0], g32=E[2,1])
        
        return E_object


    






class Metric_cartesian():

    def __init__(self, g11, g22=None, g33=None, g12=None, g13=None, g21=None, g23=None, g31=None, g32=None):

        if g11 is not None:
            self.g11 = g11
        else:
            self.g11 = '0'
        if g12 is not None:
            self.g12 = g12
        else:
            self.g12 = '0'
        if g13 is not None:
            self.g13 = g13
        else:
            self.g13 = '0'

        if g21 is not None:
            self.g21 = g21
        else:
            self.g21 = '0'
        if g22 is not None:
            self.g22 = g22
        else:
            self.g12 = '0'
        if g23 is not None:
            self.g23 = g23
        else:
            self.g23 = '0'

        if g31 is not None:
            self.g31 = g31
        else:
            self.g31 = '0'
        if g32 is not None:
            self.g32 = g32
        else:
            self.g32 = '0'
        if g33 is not None:
            self.g33 = g33
        else:
            self.g33 = '0'



    def inverse(self):

            g_ij = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
            if all(val == 0 for val in g_ij[:,0]) and all(val == 0 for val in g_ij[0,:]):
                g_ij.col_del(0)
                g_ij.row_del(0)
            if all(val == 0 for val in g_ij[:,2]) and all(val == 0 for val in g_ij[2,:]):
                g_ij.col_del(2)
                g_ij.row_del(2)
            
            gij = g_ij.inv()

            if np.shape(gij) == (3,3):
                gij_object = Metric_cartesian(g11=gij[0,0], g22=gij[1,1], g33=gij[2,2],
                                              g12=gij[0,1], g13=gij[0,2], g21=gij[1,0],
                                              g23=gij[1,2], g31=gij[2,0], g32=gij[2,1])
            if np.shape(gij) == (2,2):
                gij_object = Metric_cartesian(g11=gij[0,0], g22=gij[1,1],
                                              g12=gij[0,1], g21=gij[1,0])
            
            return gij_object



    def print(self, comp=None):

        g_ij = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in g_ij[:,0]) and all(val == 0 for val in g_ij[0,:]):
            g_ij.col_del(0)
            g_ij.row_del(0)
        if all(val == 0 for val in g_ij[:,2]) and all(val == 0 for val in g_ij[2,:]):
            g_ij.col_del(2)
            g_ij.row_del(2)
        
        if comp is None:
            print(g_ij)
        else:
            print(g_ij[comp[0], comp[1]])
    


    def cart_to_pol(self):

        g_cart = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in g_cart[:,0]) and all(val == 0 for val in g_cart[0,:]):
            g_cart.col_del(0)
            g_cart.row_del(0)
        if all(val == 0 for val in g_cart[:,2]) and all(val == 0 for val in g_cart[2,:]):
            g_cart.col_del(2)
            g_cart.row_del(2)
        
        vector1 = sm.Matrix([[x_pol], [y_pol], [z_pol]])
        vector2 = sm.Matrix([[r],[theta],[phi]])

        J_cart_to_pol = vector1.jacobian(vector2)

        g_cart_rtp = g_cart.subs(x, x_pol)
        g_cart_rtp = g_cart_rtp.subs(y, y_pol)
        g_cart_rtp = g_cart_rtp.subs(z, z_pol)

        g_pol = J_cart_to_pol.transpose() * g_cart_rtp * J_cart_to_pol
        g_pol = g_pol.expand()
        

        g_pol_object = Metric_polar(g11=g_pol[0,0], g22=g_pol[1,1], g33=g_pol[2,2],
                                    g12=g_pol[0,1], g13=g_pol[0,2], g21=g_pol[1,0],
                                    g23=g_pol[1,2], g31=g_pol[2,0], g32=g_pol[2,1])
        
        if np.shape(g_pol_object) == (3,3):
            g_pol_object = Metric_polar(g11=g_pol[0,0], g22=g_pol[1,1], g33=g_pol[2,2],
                                        g12=g_pol[0,1], g13=g_pol[0,2], g21=g_pol[1,0],
                                        g23=g_pol[1,2], g31=g_pol[2,0], g32=g_pol[2,1])
        if np.shape(g_pol_object) == (2,2):
            g_pol_object = Metric_polar(g11=g_pol[0,0], g22=g_pol[1,1],
                                        g12=g_pol[0,1], g21=g_pol[1,0])
        
        return g_pol_object
    


    def to_permittivity(self):

        gij = sm.Matrix([[self.g11, self.g12, self.g13],
                         [self.g21, self.g22, self.g23],
                         [self.g31, self.g32, self.g33]])
        
        
        if all(val == 0 for val in gij[:,0]) and all(val == 0 for val in gij[0,:]):
            gij.col_del(0)
            gij.row_del(0)
        if all(val == 0 for val in gij[:,2]) and all(val == 0 for val in gij[2,:]):
            gij.col_del(2)
            gij.row_del(2)
        
        det_sqrt = sm.sqrt(gij.det())

        E = det_sqrt * gij

        if np.shape(E) == (3,3):
            E_object = Permittivity(g11=E[0,0], g22=E[1,1], g33=E[2,2],
                                    g12=E[0,1], g13=E[0,2], g21=E[1,0],
                                    g23=E[1,2], g31=E[2,0], g32=E[2,1])
        if np.shape(E) == (2,2):
            E_object = Permittivity(g11=E[0,0], g22=E[1,1],
                                    g12=E[0,1], g21=E[1,0])

        
        return E_object



        








class Permittivity():

    def __init__(self, g11, g22=None, g33=None, g12=None, g13=None, g21=None, g23=None, g31=None, g32=None):

        if g11 is not None:
            self.g11 = g11
        else:
            self.g11 = '0'
        if g12 is not None:
            self.g12 = g12
        else:
            self.g12 = '0'
        if g13 is not None:
            self.g13 = g13
        else:
            self.g13 = '0'

        if g21 is not None:
            self.g21 = g21
        else:
            self.g21 = '0'
        if g22 is not None:
            self.g22 = g22
        else:
            self.g12 = '0'
        if g23 is not None:
            self.g23 = g23
        else:
            self.g23 = '0'

        if g31 is not None:
            self.g31 = g31
        else:
            self.g31 = '0'
        if g32 is not None:
            self.g32 = g32
        else:
            self.g32 = '0'
        if g33 is not None:
            self.g33 = g33
        else:
            self.g33 = '0'



    def visualise(self, pt_of_focus, ax_len, figsize=None):

        if figsize==None:
            fgs1=12
            fgs2=12
        else:
            fgs1=figsize[0]
            fgs2=figsize[1]

        E = sm.Matrix([[self.g11, self.g12, self.g13],
                       [self.g21, self.g22, self.g23],
                       [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in E[:,0]) and all(val == 0 for val in E[0,:]):
            E.col_del(0)
            E.row_del(0)
        if all(val == 0 for val in E[:,2]) and all(val == 0 for val in E[2,:]):
            E.col_del(2)
            E.row_del(2)

        E = E.subs(r, r_cart)
        E = E.subs(theta, theta_cart)
        E = E.subs(phi, phi_cart)

        E = E.subs(x, sm.symbols('(xg)'))
        E = E.subs(y, sm.symbols('(yg)'))
        E = E.subs(z, sm.symbols('(zg)'))

        gridx = np.linspace((pt_of_focus[0]-(ax_len/2)),(pt_of_focus[0]+(ax_len/2)),400)
        gridy = np.linspace((pt_of_focus[1]-(ax_len/2)),(pt_of_focus[1]+(ax_len/2)),400)
        xg, yg = np.meshgrid(gridx, gridy)
        zg = 0


        E_comps = []

        for i in range(np.shape(E)[0]):
            for j in range(np.shape(E)[1]):
                elem = eval(str(E[i,j]))
                if type(elem)==float or type(elem)==np.float64:
                        elem = np.full((400, 400), elem)
                if type(elem)==int:
                        elem = np.full((400, 400), 0)
                E_comps.append(elem)
                print('done comp: [{},{}]'.format(i,j))
                print(type(elem))


        fig = plt.figure(figsize=(fgs1,fgs2))

        for i in range(len(E_comps)):
            ax = fig.add_subplot(int(len(E_comps)**0.5), int(len(E_comps)**0.5), i+1)
            ax.set_aspect('equal')
            img = ax.imshow(E_comps[i],
                            extent=[(pt_of_focus[0]-(ax_len/2)), (pt_of_focus[0]+(ax_len/2)),(pt_of_focus[1]-(ax_len/2)), (pt_of_focus[1]+(ax_len/2))],
                            origin='lower',
                            cmap='RdGy')
            
        cb_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(img, cax=cb_ax)
        plt.show()





    def print(self, comp = None):

        g_ij = sm.Matrix([[self.g11, self.g12, self.g13],
                          [self.g21, self.g22, self.g23],
                          [self.g31, self.g32, self.g33]])
        
        if all(val == 0 for val in g_ij[:,0]) and all(val == 0 for val in g_ij[0,:]):
            g_ij.col_del(0)
            g_ij.row_del(0)
        if all(val == 0 for val in g_ij[:,2]) and all(val == 0 for val in g_ij[2,:]):
            g_ij.col_del(2)
            g_ij.row_del(2)
        
        if comp is None:
            print(g_ij)
        else:
            print(g_ij[comp[0], comp[1]])
  







