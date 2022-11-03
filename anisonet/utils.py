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
from collections import defaultdict

from brian2 import Equations 
from brian2 import mV, nS, pA, ms, second

from pdb import set_trace

def coord2idx(coords, pop):
    """
    Transforms the coordinates to the indices for a given population.
    
    :param coords: coordinates of n neuron 
    :type coords: numpy array of size (n,2)
    :param pop: population object 
    :type pop: Brian's NetworkGroup object
    :return: array of indicies of length N of type int
    :rtype: numpy array 

    """
    gs = int(np.sqrt(len(pop))) # gridsize
    
    coords = np.asarray(coords).reshape(-1,2)
    idxs = coords[:,1]*gs + coords[:,0]
    return idxs
    
def idx2coords(idxs, net):
    """
    Transforms the a list of indices of the given population to coordinates.
    
    :param idxs: list or array of coordinates. i.e., [(x1,y1), (x2,y2), ...]
    :type idxs: list or numpy array
    :param net: population object 
    :type net: Brian's NetworkGroup object
    :return: array of coordinates of size (N,2) of type int
    :rtype: numpy array

    """
    gs = int(np.sqrt(len(net))) # gridsize
    
    idxs = np.asarray(idxs)
    y,x = np.divmod(idxs, gs)
    coords = np.array([x,y]).T
    return coords


def aggregate_mons(sim, mon_name, SI=False):
    """
    Aggregates the indices and timings of the spiking events from disk.
    
    :param mon_name: The name of monitor of interest
    :type mon_name: str
    :return: tuple of indices and times (in ms)
    :rtype: (array of ints, array of floats)

    """
    
    data_path = sim.data_path
    name_pattern = sim.name+ '_'+ mon_name+'_*.dat'
    
    files_list = sorted(glob.glob( osjoin(data_path, name_pattern)))
    mon = {}
    for file in sorted(files_list):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
            for key, value in data.items():
                if key in mon:
                    mon[key] = np.append(mon[key], value)
                else:
                    mon[key] = value
            # ts.append(list(data['t']/ms))
            
            # idxs.append(list(data['i']))
    
    # if not SpikeMonitor, then we have to reshape the monitors
    if 'syn' in mon_name:
        for key, value in mon.items():
            if key not in ['t', 'N']:
                mon[key] = value.reshape(len(mon['t']),-1)
                
    # if SI:
    #     mon['t'] /= (1*second)
    # # idxs = np.concatenate(idxs)
    # # ts = np.concatenate(ts)

    return mon

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
                # set_trace()
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
