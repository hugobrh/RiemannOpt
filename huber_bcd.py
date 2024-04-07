#%% Prereqs

import matplotlib.pyplot as plt
import numpy as np
import autograd
import autograd.numpy as anp

import pymanopt
from pymanopt.manifolds import ComplexFixedRankSubspaceProjection,ComplexFixedRankEmbedded
from pymanopt.solvers import SteepestDescent
from helpers import hub,dhub,add_noise
from helpers import gen_dict,vec,unvec,blockshaped,row_thres,make_img
from scene_data import nx,nz,R_mp,N,dim_img,dim_scene,dx,dz,px,pz

from matplotlib.patches import Circle
targets = [(px[i],pz[i]) for i in range(len(px))]

def inds(block_size:tuple,mat_size:tuple):
    '''Construct indices list for each block'''
    
    #nb of blocks verically and horizontally
    nv  = mat_size[0] // block_size[0]
    nh  = mat_size[1] // block_size[1]
    
    #vertical and horizontal indices of upper left pts
    nvs = [i*block_size[0] for i in np.arange(nv)]    
    nhs = [i*block_size[1] for i in np.arange(nh)]
    
    #list of upper left indices
    upperleftindices = []
    for nv in nvs:
        for nh in nhs:
            upperleftindices.append([nv,nh])
    
    #create list of indices for each block
    indices = []
    for upleftind in upperleftindices:
        blockinds = []
        for j in range(block_size[0]):
           for k in range(block_size[1]):
               pt = tuple(np.array(upleftind) + np.array([j,k]))
               blockinds.append(pt)
        indices.append(blockinds)
        
    return indices

def build_psi_grad(Psis,inds):
    dictmat = []
    for i in range(len(inds)):
        mat = []
        for ind in inds[i]:
            j,k = ind
            mat.append(Psis[k][j,:])
        mat = np.vstack(mat)
        dictmat.append(mat.conj().T)
    return dictmat

def fobj_r(r,Psi_A,L,Y,c,block_size):
    ''' objective fct for r-step '''
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)       
    Es = blockshaped(Y-L-Psi_I_kron_r,*(block_size))
    Es = Es.reshape(Es.shape[0],-1)
    Es_n = anp.sqrt(anp.apply_along_axis(anp.sum,1,anp.abs(Es)**2))
    Es_hub = hub(Es_n,c)
    sum_es = anp.sum(Es_hub)
    return sum_es


def grad_fobj_r(r,Psi_A,psis_gr_mat,L,Y,c,block_size):
    ''' gradient (ie 2df/dr*) of obj fct wrt r'''
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)       
    Es = blockshaped(Y-L-Psi_I_kron_r,*(block_size))
    Es = Es.reshape(Es.shape[0],-1)
    Es_n = anp.sqrt(anp.apply_along_axis(anp.sum,1,anp.abs(Es)**2))
    Es_dhub = dhub(Es_n,c)
    Es_gr= anp.kron(Es_dhub.ravel()/Es_n.ravel(),anp.ones(Es.shape[1])) * Es.ravel()  
    grad = -psis_gr_mat @ Es_gr
    return grad

def btls(f,x,grad_fx,a = 1,c1 = 1e-4,t = 0.1,args=()):
    '''back-tracking line-search (armijo condition) '''
    while f(x - a*grad_fx,*args) > f(x,*args) - a*c1*anp.linalg.norm(grad_fx)**2:
        a = a * t
    return a

#Via Riemannian gradient descent

def JmnA_ij(shpY,A,i,j,m,n):
    if i != m:
        return 0
    else:
        return A[n,j]

#Embedded param
def create_cost_function_egrad_embedded(Y,c):
    @pymanopt.function.Callable
    def cost(u, s, vh):
        Y_hat = u @ anp. diag(s) @ vh
        return anp.sum(hub(Y - Y_hat, c))

    @pymanopt.function.Callable
    def egrad(u, s, vh):
        ''' 
        the cost function being optimized has been defined
        in terms of the low-rank singular value decomposition of X, the
        gradient returned by the autodiff backends will have three components
        and will be in the form of a tuple egrad = (df/du, df/ds, df/dv).
        '''
        gu =  anp.conjugate(autograd.grad(cost,argnum=0)(u, s, vh))
        gs =  anp.conjugate(autograd.grad(cost,argnum=1)(u, s, vh))
        gvh = anp.conjugate(autograd.grad(cost,argnum=2)(u, s, vh))
        return (gu,gs,gvh)

    return cost, egrad


#Subproj param
def create_cost_function_egrad_subproj(Y,c):
    @pymanopt.function.Callable
    def cost(u, v):
        Y_hat = u @ anp.conjugate(v).T
        return anp.sum(hub(Y - Y_hat, c))

    @pymanopt.function.Callable
    def egrad(u, v):
        ''' 
        the cost function being optimized has been defined
        in terms of the low-rank singular value decomposition of X, the
        gradient returned by the autodiff backends will have three components
        and will be in the form of a tuple egrad = (df/du, df/ds, df/dv).
        '''
        gu =  anp.conjugate(autograd.grad(cost,argnum=0)(u,v))
        gv = anp.conjugate(autograd.grad(cost,argnum=1)(u,v))
        return (gu,gv)
    
    # @pymanopt.function.Callable
    # def egrad(u,v):
    #     z = u @ anp.conjugate(v).T - Y
    #     dh = dhub(z,c)
    #     dhcj = dhub((anp.conjugate(z)),c)
    
    #     gu = anp.zeros(u.shape,dtype=u.dtype) 
    #     gv = anp.zeros(v.shape,dtype=v.dtype)   
        
    #     JU = anp.zeros((u.shape[0],v.T.shape[1],*u.shape),dtype=u.dtype)
    #     JV = anp.zeros((u.shape[0],v.T.shape[1],*v.shape),dtype=u.dtype)
        
    #     for i in range(u.shape[0]):
    #         for j in range(v.T.shape[1]):
    #             JU[i,j,i,:] = v.T[:,j]
    #             JV[i,j,j,:] = u.T[:,i]
        
    #     gu = anp.tensordot(JU,dh,axes=([0,1],[0,1]))
    #     gv = anp.tensordot(JV,dhcj,axes=([0,1],[0,1]))
    
        
    #     # guad =  anp.conjugate(autograd.grad(cost,argnum=0)(u,v))
    #     # gvad = anp.conjugate(autograd.grad(cost,argnum=1)(u,v))
    #     # print('gu-guad : ', np.linalg.norm(gu-guad))
    #     # print('gv-gvad : ', np.linalg.norm(gv-gvad))
    
    #     return (gu,gv)
    
    return cost, egrad


#algo BCD : L-step : RGD + r-step: PGD

def hub_bcd(Y,Psi,rk,c,lbd,eps=1e-3,nits=None,t_r=None,psis_gr_mat=None):
    
    U,s,Vh = anp.linalg.svd(Y)
    L = U[:,:rk] @ np.diag(s[:rk]) @ Vh[:rk]

    R = np.zeros((nx*nz,R_mp),dtype=Y.dtype)
    r = vec(R)
    
    Psi_A = anp.vstack(anp.hsplit(Psi,N))
    
    block_size = (1,1)
    
    #sub-dictionaries    
    if psis_gr_mat is None:
        indices = inds(block_size,Y.shape)
        psis_gr = build_psi_grad(anp.hsplit(Psi,N),indices)
        psis_gr_mat = anp.hstack(psis_gr)
        del(indices,psis_gr)
    
    if t_r is None:
        do_btls = True
    else:
        do_btls = False

    if not nits==None:
        it = 0
    
    res_hist = []
    
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)

    cond=False
    while cond==False:
        
        #L-step : RGD
        manifold = ComplexFixedRankSubspaceProjection(*Y.shape, rk)
        cost, egrad = create_cost_function_egrad_subproj(Y-Psi_I_kron_r,c=c)

        # manifold = ComplexFixedRankEmbedded(*Y.shape, rk)
        # cost, egrad = create_cost_function_egrad_embedded(Y-Psi_I_kron_r,c=c)
        
        problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
        solver = SteepestDescent(maxiter=50)
        
        U,V = solver.solve(problem)        
        L = anp.array(U) @ anp.conjugate(anp.array(V)).T
        
        # U,S,Vh = solver.solve(problem)
        # L =  U @ np.diag(S) @ Vh

    
        #r-step : PGD with linesearch + acceleration 
        k = 0
        r_prev = r.copy()
        for itPGD in range(1):
            w= k/(k+3)
            s = r + w*(r - r_prev)
            r_prev = r.copy()
            
            grad_r = grad_fobj_r(r,Psi_A,psis_gr_mat,L,Y,c,block_size)
            
            if do_btls:  
                t_r = btls(fobj_r,r,grad_r,a=1,args=(Psi_A,L,Y,c,block_size))
                print(f't_r:{t_r}')
            R = row_thres(unvec(s -t_r*grad_r,R.shape),t_r*lbd)
            r = vec(R)
            
            k += 1
        print(f'r-loops: {k}')
        
        #Routines
        Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)
        res = np.sum(hub(Y-L-Psi_I_kron_r,c))/anp.linalg.norm(Y) 
        print("res: ", res, " || " , eps)
        res_hist.append(res)
        
        if not nits==None:
            cond = it >= nits
            it += 1
        else:
            cond = res <= eps
            
    plt.plot(res_hist)
    plt.show()
                
    return L,R

def plot_bcd(r,c,lbd,plt_circ=False):
    #Plot        
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(r),aspect='auto',
               extent = [0,dim_scene[0],0,dim_scene[1]])
    if plt_circ:
       for i in range(len(targets)):
           circ = Circle(xy=targets[i], radius=2*dx,
                         linewidth=1.61, edgecolor='r', facecolor='none')
           ax.add_patch(circ)
    
    ax.set_xticks(np.arange(0,dim_scene[0],0.5))
    ax.set_yticks(np.arange(0,dim_scene[1],0.5))
    ax.set_title(r"$c={},\lambda$={}".format(c,lbd))
    fig.colorbar(im)
    #plt.savefig(r"../grid_search_bcd/c{}_lbd_{}.png".format(c,lbd))
    plt.show()




#%% Data loading

if __name__ == "__main__":
    
    PSI = gen_dict("../PSI_gpr.npy")
    PSI = anp.array(PSI,dtype=np.complex64)
    
    Y_mp = anp.load('../TTW_Bscan_no_interior_walls_ricker_merged_3mm_curated_resampled.npy')
    rk = 1
    
    # #Preprocess Y to samples of zero mean and unit norm
    # Y_mp = Y_mp - Y_mp.mean(axis=0)
    # col_norms = np.linalg.norm(Y_mp,axis=0)
    # Y_mp = Y_mp / col_norms[np.newaxis:,]
    
    #Noise
    t_df = 2.1  
    snr_db = 12
    snr = 10**(snr_db/10)
    
    noise_type = 'pt'
    noise_dist = 'student'
    
    np.random.seed(4)
    
    Y_mp_noised = add_noise(Y_mp, snr, t_df,noise_type=noise_type,
                            noise_dist=noise_dist)
    Y_mp_noised = anp.array(Y_mp_noised,dtype=anp.complex64)
    Y_mp_noised = Y_mp_noised/anp.max(anp.abs(Y_mp_noised))


#%% Profiling
    
#    load_ext line_profiler
#    lprun -f hub_bcd hub_bcd(Y_mp_noised, PSI, rk=rk, c=c, lbd=lbd,t_r=1e-5, nits=20)

#%% Algo
if __name__ == "__main__":

    c   = 0.1
    lbd = 1
    
    L,R = hub_bcd(Y_mp_noised, PSI, rk=rk, c=c, lbd=lbd,t_r=1e-5, nits=20)
    img=make_img(np.array(R))
    plot_bcd(img, c=c, lbd=lbd,plt_circ=True)

