import numpy as np
from scipy.spatial.distance import cdist

def funcC(sigma, z):
    C = np.zeros(len(z), dtype='complex128')
    for i in range(len(C)):
        for k in range(len(C)):
            if i==k:
                continue
            C[i]+= sigma[i]*sigma[k] / (np.conj(z[i]) - np.conj(z[k]))
    return C

def funcSR(eiphi, sigma, a):
    SR = np.zeros(len(sigma), dtype='complex128')
    for i in range(len(sigma)):
        SR[i] = eiphi[i]*int(sigma[i]==0.5)/(4.0*a)
    return SR

def funcr(z):
    r = np.zeros((len(z), len(z)), dtype='complex128')
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                r[i,j] = a
                continue
            r[i,j] = np.abs(z[i]-z[j])

def funcpsi(sigma,z):
    psi = np.power(z/np.conj(z), sigma)
    return psi
    
def funceiphi(sigma, z, psi):
    eiphi = np.ones(len(z), dtype='complex128')
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                continue
            eiphi[i]*= np.power((z[i]-z[j])/(np.conj(z[i])-np.conj(z[j])), sigma[j]) * np.exp(2*1j*psi[i])
    return eiphi

def funcq(sigma, z, eiphi):
    q = np.zeros((len(z), len(z)), dtype='complex128')
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                continue
            q[i,j] = eiphi[i] * np.power((z[i]-z[j])/(np.conj(z[i])-np.conj(z[j])), sigma[i]-1)
    return q

def funcAF(sigma, z, q):
    AF = np.zeros(len(z), dtype='complex128')
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                continue
            AF[i] += sigma[i]*sigma[j]/(1.0-sigma[j]) * (np.conj(q[i,j])-np.power(-1, (sigma[i]+sigma[j]==1))*q[i,j])/(np.conj(z[i]) - np.conj(z[j]))
    return AF

def funcL(sigma, z):
    L = np.zeros(len(z), dtype='complex128')
    for i in range(len(z)):
        for j in range(1,len(z)):
            if i==j:
                continue
            L[i]*= sigma[i]*sigma[j] * np.log
    return L

def funcB(sigma, z, L, a):
    B = np.zeros((len(z), len(z)), dtype='complex128')
    for i in range(len(z)):
        for j in range(len(z)):
            if i==j:
                B[i,j] = sigma[i]*sigma[j] * np.log(L/a)
                continue
            B[i,j] = sigma[i]*sigma[j] * np.log(L/np.abs(z[i]-z[j]))
    return B

def get_trajectory(defects_loc, defects_charge, L, a, alpha, tMax, dt=0.01):

    times = np.arange(start=0, step=dt, stop=tMax)
    z = np.zeros((len(times), len(defects_charge)), dtype='complex128')

    for d in range(len(defects_loc)):
        z[0,d] = defects_loc[d][0]-defects_loc[d][1]*1j
    
    phis = np.zeros((len(times), len(defects_charge)))
    defects_loc = np.array(defects_loc)
    defects_charge = np.array(defects_charge)
    X = np.zeros(z.shape, dtype=float)
    Y = np.zeros(z.shape, dtype=float)
    
    for t_idx in range(1,len(times)):
        psis = funcpsi(defects_charge,z[t_idx-1])
        eiphis = funceiphi(defects_charge, z[t_idx-1], psis)
        qs = funcq(defects_charge, z[t_idx-1], eiphis)
        Cs = funcC(defects_charge, z[t_idx-1])
        SRs = funcSR(eiphis, defects_charge, a)
        AFs = funcAF(defects_charge, z[t_idx-1], qs)
        A = 2.0*Cs + alpha*(SRs-0.5*AFs)
        B = funcB(defects_charge, z[t_idx-1], L, a)
        zdot = np.linalg.solve(B,A)
        phis[t_idx-1,:] = np.angle(eiphis)
        z[t_idx] = z[t_idx-1] + zdot*dt
        xs = np.real(z[t_idx,:])
        ys = -np.imag(z[t_idx,:])
        pos = np.array(list(zip(xs,ys)))
        dists = cdist(pos,pos)
        np.fill_diagonal(dists, 10.0*a)
        where_dist = np.where(dists<a)
        if len(where_dist[0])>0:
            defects_charge[where_dist[0]] = 0.0
            remaining_defs = np.where(defects_charge!=0.0)
            if len(remaining_defs[0])==0 and times[t_idx]!=times[-1]:
                z[t_idx+1:]=z[t_idx]
            if len(remaining_defs[0])>0 and times[t_idx]!=times[-1]:
                z[t_idx+1:,where_dist[0]]=z[t_idx,where_dist[0]]
                defects_loc_ = []
                for d in remaining_defs[0]:
                    defects_loc_.append([np.real(z[t_idx,d]), -np.imag(z[t_idx,d])])
                x,y = get_trajectory(defects_loc_, defects_charge[remaining_defs[0]], L, a, alpha, tMax-times[t_idx+1],dt)
                z[t_idx+1:,remaining_defs[0]] = x[:,:]-y[:,:]*1j
            break

    for d in range(z.shape[1]):
        X[:,d] = np.real(z[:,d])
        Y[:,d] = -np.imag(z[:,d])
    
    return X, Y

def get_fixedpoint(defects_loc, defects_charge, L, a, alpha, dt=0.01):

    z = np.zeros(len(defects_charge), dtype='complex128')

    for d in range(len(defects_loc)):
        z[d] = defects_loc[d][0]-defects_loc[d][1]*1j
    
    phis = np.zeros(len(defects_charge))
    defects_loc = np.array(defects_loc)
    defects_charge = np.array(defects_charge)
    X = np.zeros(z.shape, dtype=float)
    Y = np.zeros(z.shape, dtype=float)
    
    while(True):
        psis = funcpsi(defects_charge,z)
        eiphis = funceiphi(defects_charge, z, psis)
        qs = funcq(defects_charge, z, eiphis)
        Cs = funcC(defects_charge, z)
        SRs = funcSR(eiphis, defects_charge, a)
        AFs = funcAF(defects_charge, z, qs)
        A = 2.0*Cs + alpha*(SRs-0.5*AFs)
        B = funcB(defects_charge, z, L, a)
        zdot = np.linalg.solve(B,A)
        phis = np.angle(eiphis)
        z = z + zdot*dt
        xs = np.real(z)
        ys = -np.imag(z)
        pos = np.array(list(zip(xs,ys)))
        dists = cdist(pos,pos)
        np.fill_diagonal(dists, 10.0*a)
        where_dist = np.where(dists<a)
        if len(where_dist[0])>0:
            defects_charge[where_dist[0]] = 0.0
            remaining_defs = np.where(defects_charge!=0.0)
            if len(remaining_defs[0])>1:
                defects_loc_ = []
                for d in remaining_defs[0]:
                    defects_loc_.append([np.real(z[d]), -np.imag(z[d])])
                x,y = get_fixedpoint(defects_loc_, defects_charge[remaining_defs[0]], L, a, alpha, dt)
                z[remaining_defs[0]] = x[:]-y[:]*1j
            break

    X[:] = np.real(z[:])
    Y[:] = -np.imag(z[:])
    
    return X, Y