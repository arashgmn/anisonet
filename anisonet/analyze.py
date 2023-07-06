#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
osjoin = os.path.join # an alias for convenient

import time 
import numpy as np
import pandas as pd

from scipy import sparse, signal
from scipy.interpolate import make_interp_spline

from sklearn import manifold
from sklearn.cluster import (DBSCAN, OPTICS, SpectralClustering, 
                             AgglomerativeClustering)


from brian2.units import second

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from  matplotlib.cm import ScalarMappable
#import seaborn as sns

from anisonet.utils import aggregate_mons, idx2coords
from anisonet.utils import plane2torus, torus2plane, balance_dist
# from anisonet.viz import plot_3d_clusters, plot_spline_trace

from pdb import set_trace


        
def warped_clusters(xyt, gs, cluster_alg, cluster_kw={}, n_iter=1):
    """
    The idea is to find cluster twice. Once for the given sequence, and once 
    for a spatially shifted sequence. Since the boundary conditions are 
    periodic, the labels shouldn't change. In addition, those clusters that are
    cut by a border, after a shift, have the chance to merge. The best shift is 
    displacing the space by half of the network size. 
    
    Once we have the two labels, we keep one as reference, and change the other
    one until all the labels in the second one, correspond to the first one.

    """

    xyt_c = np.copy(xyt)
    xyt_c[:,:2] = (xyt_c[:,:2] + gs//2) % gs # shifted to the center
    
    if cluster_alg=='dbscan':
        cluster = DBSCAN
    elif cluster_alg =='optics':
        cluster = OPTICS
    elif cluster_alg =='spectral':
        cluster = SpectralClustering
        
    bumps = cluster(**cluster_kw).fit(xyt)
    bumps_c =  cluster(**cluster_kw).fit(xyt_c) 
    
    labels1 = bumps.labels_
    labels2 = bumps_c.labels_
    
    labels = np.copy(labels1) # we will change labels but not labels1
    
    # TODO: not sure why iteration is needed. Seems that the same things is done all the time.
    for _ in range(n_iter):
        
        pair_label = np.unique(list(zip(labels2, labels)), axis=0) # sorts by labels2
        lab2, lab1 = pair_label.T
        
        # The following are for checking the changes within each iteration.
        # if _ == 0:
        #     lab_pre = np.copy(lab1)
        # else:
        #     assert np.sum(lab1!=lab_pre), "iter num = {}".format(_)
        #     lab_pre = np.copy(lab1)
            
        lab2_uniqs, lab2_count = np.unique(lab2, return_counts=True)        
        lab2_uniqs = lab2_uniqs [lab2_count > 1]# only consider frequent bumps
        lab2_uniqs = lab2_uniqs [lab2_uniqs > -1] # which are not noise
        
        #print(lab2_uniqs)
        #print(np.unique(lab2))
        
        # now we iterate over all all these frequent lables in the labels2, and
        # find their analogous label1 and relabel them. 
        nchanged = 0
        for lab2_uniq in lab2_uniqs:
            
            #print('unique is {}'.format(lab2_uniq))
            #print(labels[:100])
            
            lab1_anlgs = lab1 [lab2 == lab2_uniq] # angls = analogous 
            lab1_anlgs = lab1_anlgs[lab1_anlgs>-1] # only non-noise must be relabeled
            idx = np.in1d(labels, lab1_anlgs) # let's find their index
            labels[idx] = lab1_anlgs[0] # and label them all similarly
            
            nchanged += sum(idx)
        #     print('Iteration {}: The number of labels changed: {}'.format(_, sum(idx)))
        # print('='*10)    
        print('Warp iteration {}: In total {} labels changed.'.format(_, nchanged))
        # print('='*10)    
        
    # TODO: not sure why do we need to do +-1.
    nclusters, labels = np.unique(labels + 1, return_inverse=True)
    return len(nclusters), labels - 1 
 


def warped_clusters_torus(xyt, gs, cluster_alg, cluster_kw={}):
    
    scsc = plane2torus(xyt[:,:2], gs)
    t = balance_dist(xyt[:,-1], 0,2)
    t /= 1000
    # set_trace()
    scsct = np.stack((*scsc.T, t)).T
    
    # scaling time
    #scsct[:,-1] *= 1/0.0001 * 2/(gs) 
    # scsct[:,-1] -= scsct[:,-1].min()
    # scsct[:,-1] *= 2/scsct[:,-1].max() # range: 0,2
    for i in range(5):
        print("{} is between {}, {}".format(i, scsct[:,i].min(), scsct[:,i].max() ))
    
    if cluster_alg=='dbscan':
        cluster = DBSCAN
    elif cluster_alg =='optics':
        cluster = OPTICS
    elif cluster_alg =='spectral':
        cluster = SpectralClustering
    
    # fig = plt.figure()
    # ax = fig.gca()
    # for i in range(5):
    #     ax.hist(scsct[:,i], bins=20, histtype='step', label=str(i))
    # ax.legend()
    
    bumps = cluster(**cluster_kw).fit(scsct)
    
    return len(set(bumps.labels_)), bumps.labels_, scsct
    
   
def find_bumps(sim, plot=True):
    
    pop_mons = sim.get_pop_mons()
    bumps = {}
    for id_, mon in enumerate(pop_mons):
        mon_dict = aggregate_mons(sim, mon.name, SI=True)
        idxs, ts = mon_dict['i'], mon_dict['t']
        # idxs, ts = aggregate_mons(sim, mon.name)
        coords = idx2coords(idxs, mon.source)
        
        # scale ts before clustering
        # ts_s = ts - ts.min() 
        # ts_s *= 1e4  # temporal resolution (0.1 ms) must match the spatial resolution
        
        xyt = np.stack((*coords.T, ts)).T
        
        # plt.figure()
        # ax = plt.gca()
        # for i in range(3):
        #     idx_sort = np.argsort(xyt[:,i])
        #     ax.hist(np.diff(xyt[:,i][idx_sort]), bins=50, histtype='step', label=i)
        # ax.legend()
        # xyt = (xyt - xyt.min(axis=0))/(xyt.max(axis=0) - xyt.min(axis=0)) # normalize to (0,1)
        # print(xyt [:10,2])
        # print(xyt.min(axis=0), xyt.max(axis=0))
        
        
        # finding bumps
        gs = mon.source.gs # needed for warping the bumps periodically
        
        
        # GOOD FOR ISO
        # dbscan_kw = dict(eps=0.3, min_samples=30) 
        # scsct[:,-1] *= 1/0.0001 * 2/(gs) 
        # scsct[:,-1] -= scsct[:,-1].min()
        # scsct[:,-1] *= 2/scsct[:,-1].max() # range: 0,2
        
        dbscan_kw = dict(eps=.095, min_samples=200)
        optics_kw = dict(min_samples=10, max_eps=40, )
        spectral_kw = dict(n_clusters=8,)
        
        w_name = sim.name+'_w_'+2*mon.source.name
        w = sparse.load_npz(osjoin(sim.data_path, w_name+'.npz'))
        aggl_kw = dict(n_cluster=8, connectivity = w) 
        # nbumps, labels = warped_clusters(xyt, gs, 
        #                                  cluster_alg = 'dbscan', 
        #                                  cluster_kw = dbscan_kw 
        #                                  )
        
        nbumps, labels, scsct = warped_clusters_torus(xyt, gs, 
                                         cluster_alg = 'dbscan', 
                                         cluster_kw = dbscan_kw 
                                         )
        
        #bumps = DBSCAN(eps=1.23, min_samples=100).fit(xyt); 
        print('Number of clusters found: {}'.format(len(set(labels))))
        
        # if plot:
        #     plot_3d_clusters(sim, xyt, labels, mon.name)
            #viz.plot_3d_clusters(sim, scsct[:,[0,2,-1]], labels, mon.name)
            # viz.plot_3d_clusters(sim, scsct[:,[1,3,-1]], labels, mon.name)
        
        bumps[mon.source.name] = np.stack((*coords.T, ts, labels)).T
        
    return bumps
        
def compute_speed(sim, plot=True):
    
    bumps = find_bumps(sim, plot)
    
    # def warp(df):
    #     grouped = df.groupby('label')
    #     gs = int(max(df.x.max(),df.y.max()))+1 
        
    #     for name, group in grouped:
    #         if group.x.mean() > gs/2.:
    #             group.x 
            
            
        
    for pop in bumps.keys():
        
        df = pd.DataFrame( bumps[pop], columns=['x','y','t','label'])
        df = df[df.label>=0].reset_index(drop=True)
        
        # TODO: Here the issu is that because of the periodic boundary condition average
        # might endup in the middle. We should have a metric to flag whoever passes
        # the border.
        
        df1 = df.groupby(['label', 't']).agg('mean').reset_index() # find centroids
        df1['t0'] = df1.groupby('label')['t'].transform(lambda x: x-x.min()).reset_index().t   # time from emergence of cluster
        set_trace()
        # xytl = bumps[pop]
        # xytl = xytl[xytl[:,-1]>-1]
        
        ts = []
        ts0 = []
        spls = [] 
        spls0 = []
        
        #set_trace()
        for label in range(len(df.label.unique())):
            x,y,t,t0 = df[df.label==1][['x','y','t','t0']].values.T
            
            ts.append(t)
            ts0.append(t0)
            spls.append(make_interp_spline(t, np.c_[x, y]))
            spls0.append(make_interp_spline(t0, np.c_[x, y]))
        
        # if plot:
        #     t_range = np.linspace(0,4, 201)
            
        #     plot_spline_trace(sim, spls, ts, t_range, 
        #                           name='_bump_trajectory_'+pop)
            
        #     plot_spline_trace(sim, spls0, [t_-t_.min() for t_ in ts], t_range, 
        #                           name='_bump_trajectory_reset_'+pop)
            
            
        # df = pd.DataFrame( bumps[pop], columns=['x','y','t','label'])
        # df = df[df.label>=0].reset_index(drop=True)
        
        
        # find centroids
        # centroids = df.groupby(['label', 't']).agg('mean').reset_index()
        
        # # time from emergence of cluster
        # centroids.t = centroids.groupby(['label'])['t'].transform(lambda x: x-x.min()).reset_index().t
        
        # # computing velocity
        # disp = centroids.set_index('label').groupby(level=0)[['t','x','y']].diff()
        # disp = disp.rename(columns={'x':'dx', 'y':'dy', 't':'dt'}).reset_index()
        # disp['v'] = np.sqrt(disp.dx**2 + disp.dy**2) / (disp.dt + 1e-12)
        
        # # adding time to displacement dataframe
        # disp['t'] = centroids.groupby(['label'])['t'].transform(lambda x: x-x.min()).reset_index().t
        # disp['t'] = disp['t']#/(disp['t'].max())
           
            #viz.plot_bump_speed(sim, disp, mon.name)
        
        
        # CLUSTERS
        # norm = Normalize(vmin= bumps.labels_.min(), vmax= bumps.labels_.max())
        # smap = ScalarMappable(norm=norm, cmap='coolwarm') #tab20
        # colors = smap.to_rgba(bumps.labels_)
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # is_member = bumps.labels_>=0
        # ax.scatter(xyt[is_member,0], xyt[is_member,1], xyt[is_member,2], 
        #            color=colors[is_member], marker='.',s=1)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')  
        # ax.set_zlabel('time')
        
        
        # SPEED
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # sns.scatterplot(data=disp[disp.index>-1], x='t', y='v', hue='label', ax=ax)
        
def connectivity_manifold(w, ncomp=2):
    spectral = manifold.SpectralEmbedding(n_components = ncomp, 
                                          affinity='precomputed')
    return spectral.fit_transform(w)  
    
    
def find_manifold(sim, plot=True):
    from scipy.ndimage import gaussian_filter
    
    pop_mons = sim.get_pop_mons()
    
    for id_, mon in enumerate(pop_mons):
        mon_dict = aggregate_mons(sim, mon.name, SI=True)
        idxs, ts = mon_dict['i'], mon_dict['t']
        coords = idx2coords(idxs, mon.source)
        xyt = np.stack((*coords.T, ts)).T
        
        i = idxs
        j = (ts/(sim.dt/second)).astype(int)
        j -= j.min()
        
        Ni = mon.source.N
        Nj = max(j)+1
        set_trace()
        
        print(f'I want to make a sparse matrix, make it dense, and load onnectivity at {time.ctime()}')
        X = sparse.coo_matrix((np.ones_like(i), (i,j)), shape = (Ni, Nj))
        X = X.toarray()
        w = sparse.load_npz(osjoin(sim.data_path, 
                                   sim.name + '_w_'+ 2 * mon.source.name+'.npz'))
        print(f'Now I want to start fitting at {time.ctime()}')
        ward = AgglomerativeClustering(n_clusters=6, 
                                       connectivity=w, 
                                       linkage="ward").fit(X)
        print(f'Done at {time.ctime()}')
        
        labels = ward.labels_
        #bumps = DBSCAN(eps=1.23, min_samples=100).fit(xyt); 
        print('Number of clusters found: {}'.format(len(set(labels))))
        
        #if plot:
        #    plot_3d_clusters(sim, xyt, labels, mon.name)
            #viz.plot_3d_clusters(sim, scsct[:,[0,2,-1]], labels, mon.name)
            # viz.plot_3d_clusters(sim, scsct[:,[1,3,-1]], labels, mon.name)
        
    #     bumps[mon.source.name] = np.stack((*coords.T, ts, labels)).T
        
    return ward
    
def compute_autocorr(spk_trn, half=False):
    st = spk_trn.sum(axis=0)
    autocorr = signal.correlate(st - st.mean(), st - st.mean(), 'full')  # Compute the autocorrelation
    autocorr /= autocorr.max()
    lags = signal.correlation_lags(len(st), len(st), 'full')
    if half:
        autocorr = autocorr[autocorr.size//2:]
        lags = lags[lags.size//2:]
    return autocorr, lags