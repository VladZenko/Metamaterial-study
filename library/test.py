import symengine as sm

x, y, z = sm.symbols('x y z')

m = sm.Matrix([[2*x,0,0],
               [0,2*y,0],
               [0,0,z]])

