#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I have developed this module to either reproduce figures of `[1]`_ and plot 
interesting quantities. Figures are saved in the``results`` folder with proper
name (inherited from the simulation object). 

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432 

"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from brian2.units import ms
import utils 

import os
osjoin = os.path.join # an alias for convenient

def get_cax(ax):
    """
    Returns a human-friendly colorbar axis from a given ax, to be used with 
    matplotlib's colorbar method.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    cax = inset_axes(ax, width="5%", height="100%", 
                     loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1),
                     bbox_transform= ax.transAxes, borderpad=0)
    
    return cax


def plot_in_out_deg(sim):
    """
    Plots both in- and out-degree of each population within the ``sim`` object.
    
    :param sim: ``simulate`` object
    :type sim: object
    """
    for syn in sim.syns:
        src = syn.source
        trg = syn.target
        
        gs_src = np.round(np.sqrt(src.N)).astype(int)
        gs_trg = np.round(np.sqrt(trg.N)).astype(int)
        
        in_deg = syn.N_incoming_post.reshape((gs_trg,gs_trg))
        out_deg = syn.N_outgoing_pre.reshape((gs_src,gs_src))
        
        fig, axs = plt.subplots(2,2, figsize=(7,4), constrained_layout=True,
                                gridspec_kw={'height_ratios':[1,.5]})
        
        # field map & distribution of in-degrees
        m = axs[0, 0].pcolormesh(in_deg, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[0,0]))
        axs[1 , 0].hist(syn.N_incoming_post, bins=50, density=True)
        
        # field map & distribution of out-degrees
        m = axs[0, 1].pcolormesh(out_deg, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[0,1]))
        axs[1 , 1].hist(syn.N_outgoing_pre, bins=50, density=True)
        
        
        axs[1,0].set_ylabel('Probability density')
        axs[1,0].set_xlabel('In-degree '+src.name+r'$\to$'+trg.name)
        axs[1,1].set_xlabel('out-degree '+src.name+r'$\to$'+trg.name)
        
        for ax in axs[0,:]:
            ax.set_aspect('equal')
            ax.get_yaxis().set_ticks([])
        
        #plt.tight_layout()
        path = osjoin(sim.res_path, sim.name+'_degs_'+src.name+trg.name+'.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        
def plot_firing_rates(sim, suffix='', 
                      conv_size=3, wl_size=10, 
                      z_score=False,
                      overlay=True):
    """
    Plots the overal firing rates within the active monitors of the ``sim``
    objects. The firing rate is also covolved with a 2D ricker kernel to show 
    the firing rate contrast better. It is also possibly to plot z-score of the
    firing rates as well.
    
    :param sim: ``simulate`` object
    :type sim: object
    :param suffix: An optional suffix for saved figure name, defaults to ''
    :type suffix: str, optional
    :param conv_size: Ricker kernel size, defaults to 3
    :type conv_size: int, optional
    :param wl_size: wavelet size of Ricker kernel, defaults to 10
    :type wl_size: float, optional
    :param z_score: Whether or not to plot the z-score, defaults to False
    :type z_score: book, optional
    
    """
    from scipy import signal
    
    fig, axs = plt.subplots(2,len(sim.pops), figsize=(4.5*len(sim.pops), 8))
    
    if len(sim.pops)==1:
        axs=axs.reshape(-1,1)
        
    for id_, mon in enumerate(sim.mons):
        src = mon.source
        gs = np.round(np.sqrt(src.N)).astype(int)
        counts = np.asarray(mon.count).reshape((gs,gs))
        
        # a filter to convolve with
        ricker = signal.ricker(conv_size, wl_size)
        ricker = np.outer(ricker, ricker)

        counts_conv = signal.convolve2d(counts,ricker, mode='same', boundary='wrap')
        counts_conv *= counts.max()/counts_conv.max()
        
        vmax = np.max(counts)
        vmin = 0
        if z_score:
            mean, std = counts.mean(), counts.std()
            counts = (counts-mean)/std
            vmin= -3.5
            vmax= +3.5
        
        m = axs[0,id_].pcolormesh(counts, vmin=vmin, vmax=vmax, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[0,id_]))
        
        m = axs[1,id_].pcolormesh(counts_conv, vmin=vmin, vmax=vmax, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[1,id_]))
        #axs[1,id_].set_xlim(0, gs)
        #axs[1,id_].set_ylim(0, gs)
        
        title = 'Firing rate '+src.name
        if z_score:
            title+= r'(z-score $\mu$ = %.3f, $\sigma$ = %.3f)'%(mean, std)
        for ax in axs[:, id_]:
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.get_yaxis().set_ticks([])
        
        if overlay:
            phis = sim.lscp[2*src.name[-1]]['phi']
            for ax in axs[:, id_]:
                overlay_phis(phis, ax)
                
    path = osjoin(sim.res_path, sim.name+'_rates'+suffix+'.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    

def plot_firing_snapshots(sim, tmin=None, tmax=None, alpha=0.1):
    """
    Similar to ``plot_firing_rates`` but plots the firing rates within an 
    arbitrary interval between ``tmin`` and `tmax`` as a scatter plot.
    """
    
    if tmin==None:
        tmin = 0*ms
    if tmax==None:
        tmax = sim.mons[0].t[-1]
    
    
    fig, axs = plt.subplots(2, len(sim.pops), 
                            figsize=(4.5*len(sim.pops),4))
    
    if len(sim.pops)==1:
        axs=axs.reshape(-1,1)
        
    for id_, mon in enumerate(sim.mons):
        exposure_time = (mon.it[1]>=tmin) * (mon.it[1]<tmax)
        idxs = mon.it[0][exposure_time]
        coords = utils.idx2coords(idxs, mon.source)
        plt.scatter(coords[:,0], coords[:,1], alpha=alpha, marker='o', s=2)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_title(r'Spike rate snapshot $t \in $ ['+str(tmin/ms)+
                      ', '+str(tmax/ms)+'] | '+mon.source.name)    
        
        #plt.tight_layout()
        path = osjoin(sim.res_path, sim.name+'_snapshot_'+mon.source.name+'.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        
        

def plot_field(field, figpath, vmin=None, vmax=None, phis=None):
    """
    Plots a field; a landscape, firing rate, or any heatmap.
    
    :param field: field as a 2D array
    :type field: numpy array
    :param name: saving name
    :type name: str
    :param vmin: minimum value of the field, defaults to None
    :type vmin: float, optional
    :param vmax: maximum value of the field, defaults to None
    :type vmax: float, optional
    """
    
    m = plt.pcolormesh(field, vmin=vmin, vmax=vmax, shading='flat', 
                       cmap='twilight_shifted')
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(m)
    
    if phis is not None:
        print(phis.shape)
        overlay_phis(phis, ax)
        
    plt.savefig(figpath, dpi=200, bbox_inches='tight')
    plt.close()
    
    
def plot_periodicity(sim, N=10):
    for id_, key in enumerate(sim.conn_cfg.keys()):
        src, trg = key
        spop = sim.pops[src]
        tpop = sim.pops[trg]
        
        gs = int(np.sqrt(tpop.N))
        periodicity_idxs = np.random.choice(spop.N, N)
        
        posts = sim.syns[id_].j.__array__()
        pres = sim.syns[id_].i.__array__()
        for plot_id, s_idx in enumerate(periodicity_idxs):
            t_idxs = posts[pres==s_idx]
            t_coords = utils.idx2coords(t_idxs, tpop)
            s_coord = utils.idx2coords(s_idx, spop)
            
            post_cntr = (t_coords-s_coord).astype(float) # centers
            post_cntr -= np.fix(post_cntr/(gs/2)) *gs # make periodic
            
            fig, axs = plt.subplots(1, 2, figsize=(8,5))    
            
            # posts
            axs[0].scatter(t_coords[:,0],
                            t_coords[:,1], 
                            marker='v', c='b')
            
            axs[1].scatter(post_cntr[:,0],
                            post_cntr[:,1], 
                            marker='v', c='b')
            
            # pre
            axs[0].scatter(s_coord[0], s_coord[1], 
                            marker='x', c='r')
            axs[1].scatter([0],[0], marker='x', c='r')
            
            # lims
            for ax in axs:
                ax.set_xlim(-2*gs, 2*gs)
                ax.set_ylim(-2*gs, 2*gs)
                ax.set_aspect('equal')
                
            # idx borders
            axs[0].vlines([0, gs], -2*gs, 2*gs, 
                        linestyle='dashed', colors='k')
            
            axs[0].hlines([0, gs], -2*gs, 2*gs, 
                        linestyle='dashed', colors='k')
            
            axs[1].vlines([-gs/2, gs/2], -2*gs, 2*gs, 
                        linestyle='dashed', colors='k')
            
            axs[1].hlines([-gs/2, gs/2],-2*gs, 2*gs, 
                        linestyle='dashed', colors='k')
            
            
            # titles
            axs[0].set_title(str(s_idx)+' Before '+src+r'$\to$'+trg)
            axs[1].set_title(str(s_idx)+' After '+src+r'$\to$'+trg)
            
            path = osjoin(sim.res_path, sim.name+'_periodicity_'+ key + '_'+ 
                                        str(plot_id+1)+'.png'
                                        )
            plt.savefig(path, bbox_inches='tight', dpi=200)
            plt.close()
            plot_id += 1
        del post_cntr, pres, posts
        
def plot_landscape(sim, overlay=True):
    """
    Plots the intended landscape (only :math:`\\phi`) for all the populations
    set in a simulation.
    
    :param sim: ``simulate`` object
    :type sim: object
    """
    for id_, key in enumerate(sim.conn_cfg.keys()):
        phis = sim.lscp[key]['phi']
        figpath = osjoin(sim.res_path, sim.name+'_gen_phi_'+key+'.png')
        if overlay:	    
            plot_field(phis.reshape((int(np.sqrt(len(phis))), -1)), figpath, 
                       vmin=-np.pi, vmax = np.pi, 
                       phis=phis)
        else:
            plot_field(phis.reshape((int(np.sqrt(len(phis))), -1)), figpath, 
                       vmin=-np.pi, vmax = np.pi)

def plot_realized_landscape(sim):
    """
    Plots the realized landscape (only :math:`\\phi`) for all the populations
    set in a simulation.
    
    :param sim: ``simulate`` object
    :type sim: object
    """
    for id_, key in enumerate(sim.conn_cfg.keys()):
        src, trg = key
        spop = sim.pops[src]
        tpop = sim.pops[trg]
        gs_s = int(np.sqrt(spop.N)) # source pop grid size
        gs_t = int(np.sqrt(tpop.N)) # target pop grid size

        phis = np.zeros(spop.N) # containts realized phi        
        posts = sim.syns[id_].j.__array__()
        pres = sim.syns[id_].i.__array__()
        
        for _, s_idx in enumerate(sorted(set(pres))):
            t_idxs = posts[pres==s_idx]
            t_coords = utils.idx2coords(t_idxs, tpop)
            s_coord = utils.idx2coords(s_idx, spop)*gs_t/gs_s
            
            post_cntr = (t_coords-s_coord).astype(float) # centers
            post_cntr -= np.fix(post_cntr/(gs_t/2)) *gs_t # make periodic
            phis[s_idx] = np.arctan2(post_cntr[:,1].mean(), post_cntr[:,0].mean())
       
        
        figpath = osjoin(sim.res_path, sim.name + '_realized_phi_'+ key+'.png')
        plot_field(phis.reshape(gs_s, gs_s), figpath=figpath, vmin=-np.pi, vmax=np.pi)
        
        # density plot
        plt.hist(phis, bins=50, density =True)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$p(\phi)$')
        plt.title(r'Realized $\phi$ for '+key)
        plt.tight_layout()
        plt.xlim(-np.pi, np.pi)
        
        figpath = osjoin(sim.res_path, sim.name + '_realized_phi_density_'+ key+'.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=200)
        plt.close()
        del post_cntr, pres, posts
        
def plot_connectivity(sim):
    """
    Plots connectivity matrix of a given connectivity file and saves the figure
    in the ``results`` folder.
    
    :param sim: ``simulate`` object
    :type sim: object
    """
    from scipy import sparse
    
    for pathway in sim.conn_cfg.keys():
        path = osjoin(sim.data_path, sim.name+'_w_'+pathway+'.npz')
        w = sparse.load_npz(path).toarray()
    
        plt.figure()
        plt.spy(w, origin='lower', rasterized=True,)
        plt.xlabel('targets (post-synapse)')
        plt.ylabel('sources (pre-synapse)')
        plt.title(' '.join(sim.name.split('_')[:2]))
        
        figpath = osjoin(sim.res_path, sim.name+'_w_'+pathway+'.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=200)
        plt.close()

def overlay_phis(phis, ax, size=5, color='k', scale=30, **kwargs):
    gs = len(phis)
    if len(phis.shape)==1:
        gs = np.sqrt(gs).astype(int)
        phi = np.copy(phis).reshape(gs, gs)
    else:
        phi = np.copy(phis)
        
    X,Y = np.meshgrid(np.arange(gs), np.arange(gs))
    
    # we need to adjust the gating size if it is not divisible to the grid size
    adjust_needed = gs%size # 
    
    # in case size adjustment is needed, we do so by hopping up and down around
    # the given size to find the closest size which is divisible to the gs. So
    # the iterations go this way:
    #   it 1: size - 1        
    #   it 2: size + 1
    #   it 3: size - 2
    #   it 4: size + 2
    #   ...                        
    counter = 1
    while adjust_needed:
        # this spans back and forth for a good size
        size += (-1)**counter * (counter)
        if size>0:
            adjust_needed = gs%size        
        
    
    s = np.sin(phi).reshape(gs//size, size, gs//size, size).mean(axis=(1, 3))
    c = np.cos(phi).reshape(gs//size, size, gs//size, size).mean(axis=(1, 3))
    
    ax.quiver(X[::size, ::size] + size/2, Y[::size, ::size] + size/2,
              c, s, 
              color=color, scale=scale, **kwargs)
    
def plot_firing_rates_dist(sim):
    """
    Plots the average firing rate density in log scale.
    
    :param sim: ``simulate`` object
    :type sim: object
    """
    from pandas import value_counts
    fig, axs = plt.subplots(1, len(sim.pops))
    if len(sim.pops)==1:
        axs = [axs]
    
    for id_, mon in enumerate(sim.mons):
        idxs, ts = utils.aggregate_mons(sim.data_path, 
                                        sim.name +'_'+ mon.name+'_*.dat')
        T = (np.max(ts) - np.min(ts))/1000. # ts is in ms
        rates = value_counts(idxs)/T
        axs[id_].hist(rates, bins=50, density=True,)
        axs[id_].set_xlabel('Firing rate [Hz]')
        axs[id_].set_title('Population '+mon.name[-1])
        
    for ax in axs:
        ax.set_yscale('log')
    
    axs[0].set_ylabel('Probability density')
    
    path = osjoin(sim.res_path, sim.name+'_rates_desnsity.png')
    plt.savefig(path,bbox_inches='tight', dpi=200)
    plt.close()

    
def animator(fig, axs, imgs, vals, ts_bins=[]):
    """
    Makes the firing rate animation.
    """
    n_frames = len(vals[0]) # total number of frames
    
    def animate(frame_id):
        if len(imgs)>1:
            for pop_idx in range(len(imgs)):
                imgs[pop_idx].set_array(vals[pop_idx][frame_id])
        else:
            imgs[0].set_array(vals[0][frame_id])
            
        if len(ts_bins) > 0:
            fig.suptitle('%s ms' % ts_bins[frame_id])
        else:
            fig.suptitle('%s' % frame_id)
        
        return *imgs,

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)

    return anim

def plot_animation(sim, ss_dur=25, fps=10, overlay=True):
    """
    Aggregates the firing rate from disk and make an animation.
    
    .. note:: 
        This function is inspired by the GitHub repo of `[1]`_.
    
    :param sim: ``simulate`` object
    :type sim: object
    :param ss_dur: duration of each snapshot (frame) in animation in ms; 
        defaults to 50 (corresponding to 50 ms)
    :type ss_dur: int, optional
    """
    fig, axs = plt.subplots(1, len(sim.pops))
    if  len(sim.pops)==1:
        axs = [axs]
    
    ts_bins = np.arange(0, sim.net.t/ms + 1, ss_dur)
    
    field_imgs = []
    field_vals = []
    for id_, mon in enumerate(sim.mons):
        idxs, ts = utils.aggregate_mons(sim.data_path, 
                                        sim.name+ '_'+ mon.name+'_*.dat')
        gs = int(np.sqrt(sim.pops[mon.name[-1]].N))
        
        h = np.histogram2d(ts, idxs, bins=[ts_bins, range(gs**2 + 1)])[0]
        field_val = h.reshape(-1, gs, gs)
        field_img = axs[id_].imshow(field_val[0], vmin=0, vmax=np.max(field_val), 
                                    origin='lower' # to match field snapshots 
                                    )
        axs[id_].set_title('Population '+mon.name[-1])
        
        if overlay:
            phis = sim.lscp[2*mon.name[-1]]['phi']
            overlay_phis(phis, axs[id_])
        
        field_vals.append(field_val)
        field_imgs.append(field_img)
        
    anim = animator(fig, axs, field_imgs, field_vals, ts_bins)
    
    writergif = animation.PillowWriter(fps=fps) 
    path = osjoin(sim.res_path, sim.name+'_animation_rate.gif')
    anim.save(path , writer=writergif)
    plt.close()
    
