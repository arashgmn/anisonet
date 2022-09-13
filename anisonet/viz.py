#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I have developed this module to either reproduce figures of `[1]`_ and plot 
interesting quantities. Figures are saved in the``results`` folder with proper
name (inherited from the simulation object). 

The animiation function is copy-pasted from the repository of `[1]`_.

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432 

"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from brian2.units import ms
import utils 

root = './'

def get_cax(ax):
    """
    Returns a human-like colorbar axis from a given ax, to be used with 
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
        
        fig, axs = plt.subplots(1,2, figsize=(9,4), )
        m = axs[0].pcolormesh(in_deg, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[0]))
        
        m = axs[1].pcolormesh(out_deg, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[1]))
        
        axs[0].set_title('In-degree '+src.name+r'$\to$'+trg.name)
        axs[1].set_title('out-degree '+src.name+r'$\to$'+trg.name)
        
        for ax in axs:
            ax.set_aspect('equal')
            ax.get_yaxis().set_ticks([])
        #plt.tight_layout()
        plt.savefig(root +'results/'+sim.name+'_degs_'+src.name+trg.name+'.png', 
                    dpi=200, bbox_inches='tight')
        plt.close()
        
        
def plot_firing_rates(sim, suffix='', 
                      conv_size=3, wl_size=10, 
                      z_score=False):
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
            
    #plt.tight_layout()
    plt.savefig(root +'results/'+sim.name+'_rates'+suffix+'.png', 
                dpi=200, bbox_inches='tight')
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
        plt.savefig(root +'results/'+sim.name+'_snapshot_'+mon.source.name+'.png', 
                    dpi=200, bbox_inches='tight')
        plt.close()
        
        

def plot_field(field, name, vmin=None, vmax=None):
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
    
    m = plt.pcolormesh(field, vmin=vmin, vmax=vmax, shading='flat')
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(m)
    plt.savefig(root +'results/'+name+'.png', dpi=200, 
                bbox_inches='tight')
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
            
            
            plt.savefig(root +'results/'+sim.name+'_periodicity_'+ key + '_'+
                        str(plot_id+1)+'.png', bbox_inches='tight',
                        dpi=200)
            plt.close()
            plot_id += 1
            #set_trace()
        del post_cntr, pres, posts
        
def plot_landscape(sim):
    """
    Plots the intended landscape (only :math:`\\phi`) for all the populations
    set in a simulation.
    
    :param sim: ``simulate`` object
    :type sim: object
    """
    for id_, key in enumerate(sim.conn_cfg.keys()):
        phis = sim.lscp[key]['phi']
        phis = phis.reshape((int(np.sqrt(len(phis))), -1))
        name = sim.name+'_gen_phi_'+key
        plot_field(phis, name, vmin=-np.pi, vmax = np.pi)


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
       
        name = sim.name + '_realized_phi_'+ key
        plot_field(phis.reshape(gs_s, gs_s), name=name,
                   vmin=-np.pi, vmax=np.pi)
        
        # density plot
        plt.hist(phis, bins=50, density =True)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$p(\phi)$')
        plt.title(r'Realized $\phi$ for '+key)
        plt.tight_layout()
        plt.xlim(-np.pi, np.pi)
        
        name = sim.name + '_realized_phi_density_'+ key
        plt.savefig(root +'results/'+name+'.png',
                    bbox_inches='tight', dpi=200)
        plt.close()
        del post_cntr, pres, posts
        
def plot_connectivity(path):
    """
    Plots connectivity matrix of a given connectivity file and saves the figure
    in the ``results`` folder.
    
    :param path: path to the connectivity file
    :type path: str
    """
    from scipy import sparse

    if '.npz' not in path:
        path+='.npz'
        
    w = sparse.load_npz(path).toarray()
    plt.figure()
    plt.spy(w, origin='lower', rasterized=True,)
    plt.xlabel('targets (post-synapse)')
    plt.ylabel('sources (pre-synapse)')
    
    filename = path.split('/')[-1]
    filename = filename.split('.')[0]
    filename = filename[2:]
    plt.title(filename)
    
    plt.savefig(root +'results/connectivity_mat_'+filename+'.png',
                bbox_inches='tight', dpi=200)
    plt.close()


def imshow(fig, ax, im, h, ts_bins=[]):
    """
    **Note**: This function is copy-pasted from the GitHub repo of `[1]`_.
    
    It prepares the field for an animation.
    
    """
    if type(im) == list:
        frames = len(h[0])
    else:
        frames = len(h)

    def animate(ii):
        if type(im) == list:
            for idx in range(len(im)):
                im[idx].set_array(h[idx][ii])
        else:
            im.set_array(h[ii])
        if len(ts_bins) > 0:
            ax.set_title('%s ms' % ts_bins[ii])
        else:
            ax.set_title('%s' % ii)
        return im,

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)

    return anim

def plot_animation(mon_name, gs, ss_dur=100):
    """
    Aggregates the firing rate from disk and make an animation.
    
    :param mon_name: name of the desired monitor
    :type mon_name: str
    :param gs: grid size of the population (length of one side!)
    :type gs: int
    :param ss_dur: duration of each snapshot (frame) in animation in ms; 
        defaults to 100 (corresponding to 100 ms)
    :type ss_dur: int, optional
    
    """
    idxs, ts = utils.aggregate_mons(mon_name)
    
    ts_bins = np.arange(0, np.max(ts), ss_dur)
    h = np.histogram2d(ts, idxs, bins=[ts_bins, range(gs**2 + 1)])[0]
    hh = h.reshape(-1, gs, gs)
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(hh[0], vmin=0, vmax=np.max(hh))
    
    anim = imshow(fig, ax, im, hh, ts_bins)
    
    writergif = animation.PillowWriter(fps=10) 
    anim.save('results/animation.gif', writer=writergif)

if __name__=='__main__':
    plot_animation('homogeneous_Gamma_1_partial_',gs=100, ss_dur=50)
