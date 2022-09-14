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

# if __name__=='__main__':
#     import brian2 as b2
    
#     G = b2.NeuronGroup(N=100**2, model="dv/dt=1:1")
    
#     i2c = idx2coords([1,102], G)
#     c2i = coord2idx(i2c, G)
