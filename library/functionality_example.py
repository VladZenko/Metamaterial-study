import lib

g11 = '1/(1-1/r)'
g22 = 'r**2'
g33 = 'r**2*sin(theta)**2'

g_ij = lib.Metric_polar(g11, g22, g33)
#g_ij.print()

gij = g_ij.inverse()
#gij.print()




gij_cart = gij.pol_to_cart()
#gij_cart.print()

E = gij_cart.to_permittivity()
#E.print(comp=[0,0])
E.visualise(pt_of_focus=[0,0], ax_len=6)

#E_cart = gij_cart.to_permittivity()
#E.print()
#E_cart.visualise(pt_of_focus=[0,0], ax_len=6)

"""gij = g_ij.inverse()

gij_cart = gij.pol_to_cart()

E_cart = gij_cart.to_permittivity()

E_cart.visualise()"""