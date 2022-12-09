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

import brian2 as b2
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


def stimulator(sim, stim_cfgs):
    stims = {}
    for stim_id, stim_cfg in stim_cfgs.items():
        domain = stim_cfg['domain']
        ampli = stim_cfg['domain'] 
        pop = stim_id.strip('_')[0] # assuming ids like I_1, I_2
        gs = sim.pops[pop].gs 
        
        # finding domain
        if domain['type']=='random':
            idxs = np.random.randint(0, gs**2, round(gs**2 * domain['p']))
        else:
            if domain['type']=='xy':
                x = np.arange(domain['x_min'], domain['x_max'])
                y = np.arange(domain['y_min'], domain['y_max'])
                x,y = np.meshgrid(x,y)
                coords = np.array(list(zip(x.ravel(),y.ravel())))
            
            elif domain['type']=='r':
                coords = []
                for x in range(-round(domain['r'])-1, round(domain['r'])+1):
                    for y in range(-round(domain['r'])-1, round(domain['r'])+1):
                        if x**2 + y**2 <= domain['r']**2:
                            coords.append( [(x + domain['x0'])%gs, 
                                            (y + domain['y0'])%gs] ) 
                coords = np.array(coords).reshape(-1,2)
            
            idxs = coord2idx(coords, sim.pops[pop])
            

        # finding amplitude
        if stim_cfg['type']=='const':
            I_stim = stim_cfg['I_stim']*pA
            # b2.TimedArray([stim_cfg['I_stim']]*pA,
            #                                dt=sim.pops[pop].clock.dt)
        else:
            raise NotImplementedError('Only constant stimulation is supported now.')
        
        stims[stim_id] = {'idxs': idxs, 'I_stim': I_stim}
    
    return stims


def get_line_idx(x0, y0, c, pop, eps=2):
    xs = np.arange(0, pop.gs//2) #only half of plane will be modified
    
    coords = []
    for x in xs:
        y_center = int(round(x*c))
        for y in range(y_center-eps-1, y_center+eps+1):
            if y**2 <= eps**2:
                coords.append([(x+x0) % pop.gs, (y+y0) % pop.gs])
    coords = np.unique(coords, axis=0)
    idxs = coord2idx(coords, pop)
    return idxs, coords
    
    
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


def make_circular(r, r_max):
    return 2*np.pi*r/r_max

def make_planar(angle, r_max):
    return angle/(2*np.pi) * r_max
    
def plane2torus(p_coords, gs, method='lin'):
    #set_trace()
    # coords must have the shape (n_coords,2)
    assert p_coords.shape[1] == 2 # shape must be 
    
    t_coords = np.zeros((p_coords.shape[0],4), dtype=float)
    phi, psi = make_circular(p_coords, gs).T
    
    if method=='tri':
        # morphing x
        t_coords[:,0] = np.sin(phi)
        t_coords[:,1] = np.cos(phi)
        
        # morphing y
        t_coords[:,2] = np.sin(psi)
        t_coords[:,3] = np.cos(psi)
    
    elif method=='lin':
        for id_, ang in enumerate([phi, psi]):
            # Sin component
            choice0 = ang/(np.pi/2)     
            choice1 = -ang/(np.pi/2) + 2
            choice2 = ang/(np.pi/2) - 4
            
            index = np.zeros(ang.shape, dtype=int)
            index[ang <= np.pi/2] = 0
            index[(np.pi/2 < ang) & (ang <= 3*np.pi/2)] = 1
            index[ang > 3*np.pi/2] = 2
            
            t_coords[:,2*id_] = np.choose(index, [choice0, choice1, choice2])
                
            # cos component
            choice0 = -ang/(np.pi/2) + 1     
            choice1 = +ang/(np.pi/2) - 3
            
            index = np.zeros(ang.shape, dtype=int)
            index[ang <= np.pi] = 0
            index[ang > np.pi] = 1
            
            t_coords[:,2*id_ +1] = np.choose(index, [choice0, choice1])
            
    else:
        raise NotImplementedError("At the moment only triangulumetric and linear morphing is possible.")
    
    return t_coords


def torus2plane(t_coords, gs, method='tri'):
    # coords must have the shape (n_coords,4)
    assert t_coords.shape[1] == 4 # (s_x, c_x, s_y, c_y)
    
    p_coords = np.zeros((t_coords.shape[0],2), dtype=float)
    
    if method=='tri':
        phi = np.arctan2(t_coords[:,0], t_coords[:,1])
        psi = np.arctan2(t_coords[:,2], t_coords[:,3])
        
        p_coords[:,0] = make_planar(phi, gs)
        p_coords[:,1] = make_planar(psi, gs)
    else:
        raise NotImplementedError("At the moment only triangular morphing is possible.")
    
    return np.round(p_coords).astype(int)
    
        
def balance_dist(t0, t_min=0, t_max=1):
    t = t0-t0.min()
    t /= t.max()
    
    percents = np.linspace(0,1, len(t))
    sorted_idx = np.argsort(t)
    
    for idx, val in enumerate(percents):
        t[sorted_idx[idx]] = val
        
    t = t_min + (t_max - t_min) * t
    # n_quantiles = len(set(t))
    # quant_size = len(t)//n_quantiles 
    # print(n_quantiles, quant_size)
    
    # for quant_idx, quant_val in enumerate(np.linspace(t_min, t_max, n_quantiles+1)):
    #     t[ sorted_idx[quant_idx*quant_size : (quant_idx+ 1)*quant_size] ] = quant_val
    
    return t
    
    # sorted_idx = np.argsort(phis)
    # max_val = gs * 2
    # idx = len(phis) // max_val
    # for ii, val in enumerate(range(max_val)):
    #     phis[sorted_idx[ii * idx:(ii + 1) * idx]] = val
    # phis = (phis - gs) / gs
    
    # # to push between -pi and pi
    # phis -= np.min(phis)
    # phis *= 2*np.pi/(np.max(phis)+1e-12)
    # phis -= np.pi

def get_anisotropic_U(sim, syn_name, Umax):
    syn = sim.syns[syn_name]
    conn_cfg = sim.conn_cfg[syn_name]
    lscp = sim.lscp[syn_name]['phi']
    lscp = Umax*(lscp- lscp.min())/(lscp.max()-lscp.min())
    
    Us = np.zeros(len(syn))
    for idx_pre in sorted(set(sim.syns[syn_name].i)):
        syn_idx = syn["i=={}".format(idx_pre)]._indices()
        U_mean = lscp[idx_pre]
        alpha = 2
        beta = alpha*(1./(U_mean+1e-12) - 1)
        Us[syn_idx] = np.random.beta(alpha, beta, size= len(syn_idx))
    return Us

def get_post_rel_locs(syn, pre_idx):
    s_loc = idx2coords(pre_idx, syn.source).astype(float)
    s_loc*= syn.target.gs/syn.source.gs*1.
    s_loc = np.round(s_loc).astype(int)
    
    t_locs = get_post_locs(syn, pre_idx) - s_loc
    t_locs =  (t_locs + syn.target.gs/2) % syn.target.gs - syn.target.gs/2
    return t_locs

def get_post_idxs(syn, pre_idx):
    syns_pre = syn["i=={}".format(pre_idx)]._indices()
    return syn.j[syns_pre]

def get_post_locs(syn, pre_idx):
    post_idxs = get_post_idxs(syn, pre_idx)
    return idx2coords(post_idxs, syn.target)

def pre_loc2post_loc_rel(t_locs, s_loc, gs):
    tmp = t_locs - s_loc
    return (tmp +gs/2) % gs - gs/2
    