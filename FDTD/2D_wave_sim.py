import numpy as np
import matplotlib.pyplot as plt
import time



eps0 = 8.85418782e-12
mu0 = 1.25663706e-6
c0 = 1/np.sqrt(eps0*mu0)
imp0 = np.sqrt(mu0/eps0)


imax = 500
isource = 250
jmax = 500
jsource = 250
nmax = 100

# Define initial field values to be transformed
Ex = np.zeros((imax,jmax))
Ey = np.zeros((imax,jmax))
Hz = np.zeros((imax,jmax))
Ex_prev = np.zeros((imax,jmax))
Ey_prev = np.zeros((imax,jmax))
Hz_prev = np.zeros((imax,jmax))


lambda_min = 350e-9
dx = lambda_min/20
dy = lambda_min/20
dt = dx/c0

eps = eps0
    


fig, ax = plt.subplots(figsize=(12,8))






def Source_Function(t):
    lambda_0 = 550e-9
    w0 = 2*np.pi*c0/lambda_0
    tau = 30
    t0 = tau*3
    

    return np.exp(-(t-t0)**2/tau**2)


Ex_lst = []


for n in range(nmax):

    Hz[imax-1,jmax-1] = Hz_prev[imax-2,jmax-2]
    #Hz[imax-1,jmax-1] = Hz_prev[imax-1,jmax-2]

    for i in range(imax-1):
        for j in range(jmax-1):
            Hz[i,j] = Hz_prev[i,j] + dt/(dy*mu0)*(Ex[i,j+1]-Ex[i,j])\
                      - dt/(dx*mu0)*(Ey[i+1,j+1]-Ey[i,j])
            Hz_prev[i,j] = Hz[i,j]
    
    Hz[isource-1,jsource-1] -= Source_Function(n)/imp0
    Hz_prev[isource-1,jsource-1] = Hz[isource-1,jsource-1]


    Ex[0,0] = Ex_prev[1,1]
    Ey[0,0] = Ey_prev[1,1]

    for i in range(1,imax):
        for j in range(1,jmax):
            Ex[i,j] = Ex_prev[i,j] + dt/(dy*eps)*(Hz[i,j]-Hz[i,j-1])
            Ex_prev[i,j] = Ex[i,j]

            Ey[i,j] = Ey_prev[i,j] + dt/(dx*eps)*(Hz[i,j]-Hz[i-1,j])
            Ey_prev[i,j] = Ey[i,j]


    
    Ex[isource, jsource] += Source_Function(n+1)
    Ex_prev[isource, jsource] = Ex[isource, jsource]

    Ey[isource, jsource] += Source_Function(n+1)
    Ey_prev[isource, jsource] = Ey[isource, jsource]
    

    print(np.count_nonzero(Ex))

    if n%10==0:
        Ex_lst.append(Ex)

    """
    if n%10==0:     
        ax.clear()
        ax.contourf(Ex)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        plt.pause(0.01)
    """
    
for frame in range(len(Ex_lst)):     
        ax.clear()
        ax.imshow(Ex_lst[frame])
        ax.set_ylim(0,2)
        ax.set_xlim(0,2)
        plt.pause(0.01)
#ax.imshow(Ex_lst[-1])
plt.show()






