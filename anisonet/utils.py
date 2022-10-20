#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These utilitiy functions transform the indicies of neurons to their coordinates
back and forth. A useful little script to have.
"""
import glob
import os
osjoin = os.path.join # an alias for convenient

import pickle
import numpy as np
from brian2 import Equations 
from brian2 import mV, nS, pA, ms

root = './'

def coord2idx(coords, pop, dim=2):
    """
    Transforms the coordinates to the indices for a given population.
    
    :param coords: coordinates of n neuron 
    :type coords: numpy array of size (n,2)
    :param pop: population object 
    :type pop: Brian's NetworkGroup object
    :return: array of indicies of length N of type int
    :rtype: numpy array 

    """
    gs = pop.gs # gridsize
    
    coords = np.array(coords).reshape(-1, dim)
    
    idxs = np.zeros(coords.shape[0])
    for d in range(dim):
        idxs += coords[:,d] * gs**d
    # idxs = coords[:,1]*gs + coords[:,0]
    return idxs.astype(int)
    
def idx2coords(idxs, pop, dim=2):
    """
    Transforms the a list of indices of the given population to coordinates.
    
    :param idxs: list or array of coordinates. i.e., [(x1,y1), (x2,y2), ...]
    :type idxs: list or numpy array
    :param net: population object 
    :type net: Brian's NetworkGroup object
    :return: array of coordinates of size (N,2) of type int
    :rtype: numpy array

    """
    gs = pop.gs # gridsize
    
    idxs = np.array(idxs)
    coords = np.zeros((dim, len(idxs)))
    
    devid = np.copy(idxs)
    for d in range(dim):
        devid, coords[d,:] = np.divmod(devid, gs)
        
    # y,x = np.divmod(idxs, gs)
    # coords = np.array([x,y]).T
    return coords.T.astype(int)


def aggregate_mons(data_path, name_pattern):
    """
    Aggregates the indices and timings of the spiking events from disk.
    
    :param mon_name: The name of monitor of interest
    :type mon_name: str
    :return: tuple of indices and times (in ms)
    :rtype: (array of ints, array of floats)

    """
    
    files_list = sorted(glob.glob( osjoin(data_path, name_pattern+'*')))
    idxs = []
    ts = []
    for file in sorted(files_list):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            idxs.append(list(data['i']))
            ts.append(list(data['t']/ms))
    
    idxs = np.concatenate(idxs)
    ts = np.concatenate(ts)
    
    del files_list
    return idxs, ts


def phase_estimator(idxs, ts, dt):
    t = np.linspace(0, ts.max(), int(ts.max()//dt) + 1)
    phis = np.zeros(shape= (len(set(idxs)), len(t)))
    
    for num, idx in enumerate(sorted(set(idxs))):
        phi = np.zeros_like(t)
        
        t_spk = sorted(ts[idxs == idx])
        
        if len(t_spk)>1:
            t_dif = np.diff(t_spk)
            
            head = int(t_spk[0]//dt)
            
            for chunk_id, dif in  enumerate(t_dif):
                #set_trace()
                chunk_len = int(dif//dt)
                phi[ head: head + chunk_len] = np.linspace(0, 2*np.pi, chunk_len) 
                head += chunk_len
            
            phis[num] = phi
    
    return t, phis

def estimate_order_parameter(phis, k=None):

    if k==None:
        k = np.ones(phis.shape[0])
    else:
        assert len(k)==phis.shape[0]
        
    R = np.average(np.exp(1j*phis), weights=k, axis=0)
    return np.abs(R), np.angle(R)

# if __name__=='__main__':
#     import brian2 as b2
    
#     G = b2.NeuronGroup(N=100**2, model="dv/dt=1:1")
    
#     i2c = idx2coords([1,102], G)
#     c2i = coord2idx(i2c, G)
