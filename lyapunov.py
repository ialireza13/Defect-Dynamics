from functions import get_first_collision, get_trajectory, random_defects
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

if __name__ ==  '__main__': 

    L=1e5
    l=10
    N=3
    bias = (L-l)/2.0
    alpha=10.0
    a = 0.8*1e-1
    tMax = 10
    ens = 5
    dt = 0.1
    pert = 1e-10
    
    disloc = np.zeros((ens, N, int(tMax/dt)))

    for i in tqdm(range(ens)):

        def_loc, def_sigma = random_defects(N, l)
        def_loc = def_loc+bias
        def_sigma = np.ones(def_sigma.shape)*0.5
        while(True):
            try:
                X, Y, t = get_trajectory((def_loc, def_sigma, L, a, alpha, tMax, dt))
                noise = np.ones(def_loc.shape)*np.random.choice([pert, -pert])
                perturned_loc = def_loc+noise
                X1, Y1, t = get_trajectory((perturned_loc, def_sigma, L, a, alpha, tMax, dt))
                break
            except:
                continue
        for n in range(N):
            d1 = (X[:,n] - X1[:,n])**2
            d2 = (Y[:,n] - Y1[:,n])**2
            d = np.sqrt(d1+d2)

            disloc[i, n, :] = d
            
    np.save('disloc_dist.npy', disloc)