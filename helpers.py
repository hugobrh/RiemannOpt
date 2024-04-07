#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:26:32 2023

@author: hugobrehier
"""

import autograd.numpy as anp
import numpy as np
from scene_data import nx,nz,d,zoff,M,N,recs,R_mp,w,xw_l,xw_r,dim_img,dim_scene
from delay_propagation import (delai_propag,delai_propag_interior_wall,
                                         delai_propag_ttw,delai_propag_wall_ringing)
ratio_dims = dim_scene/dim_img

import itertools,os

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s_(%d).%s" % (basename, next(c), ext)
    return actualname
    
def vec(Y):
    return Y.ravel('F')

def unvec(v,dims:tuple): 
    return v.reshape(dims, order='F')

def csgn(X):
    return anp.where(X!=0,X/anp.abs(X),0)

def hard_thres(Y,l):
    return Y * np.where(np.abs(Y)>=l,1,0)

def soft_thres(Y,l):
    return csgn(Y) * anp.maximum(0,anp.abs(Y) - l)

def svd_thres(Y,l):
    U, s, Vh = anp.linalg.svd(Y, full_matrices=False)
    return U @ anp.diag(soft_thres(s, l)) @ Vh

def row_thres(U,l):
    '''Row-wise Thresholding (mixed l2/l1 norm proximal)'''
    nU = anp.sqrt(anp.diag(U @ U.conj().T))
    Ut = (anp.maximum(0,1-(l/nU)) * U.T).T
    return Ut

def hub(x,c):
    return ((anp.abs(x)<=c)*0.5*anp.abs(x)**2 + (anp.abs(x)>c)*c*(anp.abs(x)-0.5*c))
    
def dhub(x,c):
    return anp.where(anp.abs(x)>c,c*csgn(x),x)

def prox_huber(x,c,a):
    return anp.where(anp.abs(x)<=c*(a+1),x/(a+1),x-c*a*csgn(x))


def prox_huber_norm(X,c,a):
    """proximal of the a-scaled Huber function composed with norm, of threshold c and matrix argument X """
    nX = anp.linalg.norm(X)
    #cast nan to zero this wqy so it can be compiled by jax
    return anp.where(anp.abs(X/nX) < anp.inf, prox_huber(nX, c, a) * X/nX, anp.zeros(X.shape))


def wall_returns(fc,B,M,N,d):
    fs = np.linspace(fc-B/2,fc+B/2,M)
    sig = (1/((2*d)**2)) * np.exp(-1j*2*np.pi*fs*d/3e8)
    return np.vstack(N*[sig]).T


def make_img(R):
    r_vec = vec(R)
    #Unvectorize solution
    r_mat = unvec(r_vec,(nx*nz,-1))
    
    #combine all R sub-images 
    r_comb = np.zeros(nx*nz,dtype=np.complex64)
    for pix in range(nx*nz):
        r_comb[pix] = np.linalg.norm(r_mat[pix,:])
    
    #Unvectorize to form image
    r = unvec(r_comb,(nx,nz))
    r = np.rot90(r)
    
    return r

def add_noise(y_mp,snr,t_df,noise_type,noise_dist):
    
    M,N = y_mp.shape
    P_y = np.sum(np.abs(y_mp)**2)/(M*N)
    sig_n = np.sqrt(P_y/snr)  
    
    cgn = np.random.randn(M,N)/np.sqrt(2) + 1j*np.random.randn(M,N)/np.sqrt(2)
    cgn = sig_n*cgn

    if noise_type=='pt':
        if noise_dist=='gaussian':
            text = 1    
        
        elif noise_dist=='student':
            x = np.random.gamma(shape=t_df/2,scale=2,size= (M,N))
            text = t_df/x
        
        elif noise_dist=='k':
            text = np.random.gamma(shape=t_df,scale=1/t_df,size= (M,N))
        
        else:
            raise ValueError("check noise dist")

        y_res = y_mp + np.sqrt(text)*cgn
        
    elif noise_type=='col':
        if noise_dist=='gaussian':
            text = np.ones(N)
        
        elif noise_dist=='student':
            x = np.random.gamma(shape=t_df/2,scale=2,size= N)
            text = t_df/x
        
        elif noise_dist=='k':
            text = np.random.gamma(shape=t_df,scale=1/t_df,size=N)
        
        else:
            raise ValueError("check noise dist")

        y_res = y_mp + cgn @ np.diag(np.sqrt(text))
    
    else:
        raise ValueError("check noise type")

    return y_res

#LRA with least squares via SVD
def lra_frob_svd(Y,r):
    '''
    Low rank approx in frobenius norm via SVD (eckart young thm).
    '''
    U,s,Vh = np.linalg.svd(Y)
    return U[:,:r] @ np.diag(s[:r]) @ Vh[:r]


def lra_hub_prox(Y,c,mu,gam,tol,maxits):
    '''
    Low rank approx in huuber norm via proximal + decoupling L=M
    '''
    L = np.zeros(Y.shape,dtype=Y.dtype)
    M = np.zeros(Y.shape,dtype=Y.dtype)
    U = np.zeros(Y.shape,dtype=Y.dtype)
    
    it = 0
    e = tol + 1
    
    print(f'iteration: error')

    while (e>tol) and (it < maxits):
            
        L = prox_huber(M+U/gam - Y, c, 1/gam) + Y
        M = svd_thres(L - U/gam, mu/gam)
        U = U + gam * (M-L)
        
        it += 1
        e  = np.sum(hub(Y-L,c)) / np.sum(hub(Y,c))
        
        print(f'{it}: {e}')
        
    return L
    
    
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

#Multipath dictionary
def gen_dict(path):
    try:
        PSI = np.load(path)
    except FileNotFoundError:
        PSI = None
        print('Saved dictionary PSI not found')
        
    
    if not PSI is None:
        print("PSI already loaded from hard drive")
    else:
        
        PSI = []
        
        #create one dico of delay for each SAR position
        for n in range(N):
            delais_simple_ttw = np.zeros([nx,nz],dtype=np.complex64)
            points = np.zeros([nx,nz,2])
            
            for i in range(nx):
                for j in range(nz):
                    points[i,j] = np.array([i,j]) * ratio_dims
                    if points[i,j][1] <= (zoff+d):
                        delais_simple_ttw[i,j] = delai_propag(points[i,j],recs[n,:])
                    else:
                        _,delais_simple_ttw[i,j] = delai_propag_ttw(points[i,j],recs[n,:])
                        
            #create one dico of delay for each multipath            
            for r in range(R_mp):
                psi_r = np.zeros([M,nx*nz],dtype=np.complex64)
                
                tau = np.zeros([nx,nz],dtype=np.complex64)    
                for i in range(nx):
                    for j in range(nz):
                        if points[i,j][1] <= (zoff+d): #zone avant mur
                            tau[i,j] = delais_simple_ttw[i,j]
                        else: 
                            if r == 0:
                                tau[i,j] = delais_simple_ttw[i,j]
                            elif r == 1:
                                tau[i,j] = delais_simple_ttw[i,j]/2 + delai_propag_wall_ringing(points[i,j],recs[n,:],1)
                            elif r == 2:
                                tau[i,j] = delais_simple_ttw[i,j]/2 + delai_propag_interior_wall(points[i,j],recs[n,:],xw_r)
                            elif r == 3:
                                tau[i,j] = delais_simple_ttw[i,j]/2 + delai_propag_interior_wall(points[i,j],recs[n,:],xw_l)   
                tau = vec(tau)
                for m in range(M):
                    for pos in range(nx*nz):
                        psi_r[m,pos] = np.exp(-1j*w[m]*tau[pos])
                
                if r==0:
                    Psi_n = psi_r
                else:
                    Psi_n = np.hstack([Psi_n,psi_r])
            
            PSI.append(Psi_n)
        PSI = np.hstack(PSI)
        np.save(path,PSI)
        
    return PSI


