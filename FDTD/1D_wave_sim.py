import numpy as np
import matplotlib.pyplot as plt
import time



eps0 = 8.85418782e-12
mu0 = 1.25663706e-6
c0 = 1/np.sqrt(eps0*mu0)
imp0 = np.sqrt(mu0/eps0)


jmax = 500
jsource = 10
nmax = 750

# Define initial field values to be transformed
Ex = np.zeros(jmax)
Hz = np.zeros(jmax)
Ex_prev = np.zeros(jmax)
Hz_prev = np.zeros(jmax)

lambda_min = 350e-9
dx = lambda_min/20
dt = dx/c0

eps = eps0
    


fig, ax = plt.subplots(figsize=(12,8))






def Source_Function(t):
    lambda_0 = 550e-9
    w0 = 2*np.pi*c0/lambda_0
    tau = 30
    t0 = tau*3
    

    return np.exp(-(t-t0)**2/tau**2)*np.sin(w0*t*dt)




for n in range(nmax):

    #Hz[jmax-1] = Hz_prev[jmax-2]

    for j in range(jmax-1):
        Hz[j] = Hz_prev[j] + dt/(dx*mu0)*(Ex[j+1]-Ex[j])
        Hz_prev[j] = Hz[j]
    
    Hz[jsource-1] -= Source_Function(n)/imp0
    Hz_prev[jsource-1] = Hz[jsource-1]



    #Ex[0] = Ex_prev[1]

    for j in range(1,jmax):
        Ex[j] = Ex_prev[j] + dt/(dx*eps)*(Hz[j]-Hz[j-1])
        Ex_prev[j] = Ex[j]
    
    Ex[jsource] += Source_Function(n+1)
    Ex_prev[jsource] = Ex[jsource]
    

    if n%10==0:     
        ax.clear()
        ax.plot(Ex, color='blue')
        ax.set_ylim(-1,1)
        plt.pause(0.01)
    


plt.show()






