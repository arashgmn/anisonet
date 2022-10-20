#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module controls the inhomogeneity and anistropy of the synaptic 
connections. The probability of stablishing a connection from the presynapse at
location :math:`\\textbf r` to location :math:`\\textbf r'`, in general depends
on both locations. We model connectivies that can be broken down this into 
radial and angular terms:
    
.. math::
	w(\\textbf r, \\textbf r') = f(|\\textbf r - \\textbf r'|) g(\phi)

The function :math:`f` controls the extent of spatial homogeneity and is assumed to 
have a peak at the center. Function :math:`g` encodes anisotropy or angular 
bias in forming connections. 

There are many ways to model such dependence. We implemented two methods: a
displacement-based (`shift`) and a rotation-based (`rotate`) one which are 
explained in details below.

**Inhomogeneity**: Radial connectivity profile can enforce locality. The following profiles are
supported:
    
    #. Gaussian
    #. Gamma distribution
    #. uniform (homogeneous)

using which we sample the distance (radius) to the postsynapse. Additionally,
in cases where autapse is not allowed, we symmetrically shift the aformentioed 
PDFs such that distances less than a controllable ``gap`` are improbable. 

**Anisotropy**: We attribute an angle :math:`\phi` to each neuron that specifies its angular
bias (a.k.a. anisotropy landscape) by the following strategies:
    
    #. uniform (isotropic)
    #. constant or homogenous
    #. Perlin-noise-based: A unifromly random method with spatioal correlation


Having the landscape specified, we now explain how connections 
are actually formed.

Enforcing anisotropy
====================
Anisotropy is induced in two major steps. The first one, which is shared among
all methods, is generating an *iso*-tropic set of posible postsynaptic locations.
based on the radius drew before. These locations make an angle-independent 
presynapse-centric point cloud. The next step natually is to induce anisotropy
in this cloud. The API accepts the following values which are explained below:
    
    #. ``shift``: A displacement-based method
    #. ``squeeze-rotate``: A rotation-based method that first squeezes the point
       cloud
    #. ``positive-rotate``: A rotation-based method that constraints the point 
       cloud to positive x values before rotation. Thus, differentiates between 
       angle :math:`\\theta` and :math:`2\pi - \\theta`. 
    #. ``positive-squeeze-rotate``: combination of last two methods.
     
Dispalcement-based
~~~~~~~~~~~~~~~~~~
This method is similar to `[1]`_ and works as follows. To enforce 
anisotropy, this point cloud is shifted be a displacement vector determined by
a length :math:`r > 0`  and angle  :math:`\phi` (the landscape). We keep this 
length constant for all neurons (although it can be very well change too).

Rotation-based
~~~~~~~~~~~~~~
In this method we generate a circular point cloud as before. However, instead 
of displacing, we first squeeze this point cloud and then rotate this elongated 
point could according to the angle provided in the landscape. Squeezing is 
controlled by the :math:`r > 0` property of the landscape -- now interpreted as
the `reshpae` factor: The x-axis of the point cloud is up-scaled by :math:`1+r`
while the y-axis is down-scaled by the same factor (so that the area stays 
constant). Then the ellipse is rotated along the z-axis. Note that for these
method, complementary angles lead to the identical distributions.


Positive rotation
~~~~~~~~~~~~~~~~~
To discern between complementary angles :math:`\\theta` and :math:`\\theta' = 2\pi - \\theta` 
this method first coverts all the locations with the negative-x to positive ones.
Then, it rotates the (now half-circular) point cloud without squeezign it.

Squeeze positive rotation
~~~~~~~~~~~~~~~~~~~~~~~~~
Combines the two method above by first converting to positive x values, and 
then squeezing followed by rotating.


Other methods can be exercised as well.
    
.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
.. _[3]: https://doi.org/10.1523/ENEURO.0348-16.2017
.. _[4]: http://epubs.siam.org/doi/10.1137/030600040
.. _here: https://github.com/babsey/spatio-temporal-activity-sequence/blob/bb82c7d4d3b2f85e9d2635a5479d0478868d33bb/scripts/lib/connection_matrix.py#L28

"""

import numpy as np

# def draw_post_syns(s_coord, ncons,
#                   srow, scol, trow, tcol, 
#                   profile, shift={}, 
#                   self_link= False, recurrent=True, gap=5,
#                   p_WS = 0.37):
#     """Samples indices of the source neuron `s_coord` in the target network 
#     according to a location-dependent profile. The profile by default is 
#     localized on the location of the source neuron. Yet, it is possible to 
#     shift to any location relative to the source neuron on the target network. 
#     The location of the source neuron on the target network will be 
#     interpolated if the size of source and target networks don't match.

#     Periodic boundary condition is applied.

#     :param s_coord: source coordinate index as a tuple of ints
#     :type s_coord: tuple
#     :param ncons: number of out-links
#     :type ncons: int
#     :param srow: number of rows in source network
#     :type srow: int
#     :param scol: number of columns in source network
#     :type scol: int
#     :param trow: number of rows in target network
#     :type trow: int
#     :param tcol: number of columns in target network
#     :type tcol: int
#     :param profile: connectivity configuration. Please refer to 
#         :ref:`configs:Profile` for details.
#     :type profile: dict
#     :param shift: shift configuration Please refer to 
#         :ref:`configs:Anisotropy` for details.
#     :type shift: dict
#     :param self_link: Is self-link permitted; Matter only if source and target
#         populations are the same.
#     :type self_link: bool
#     :param recurrent: Whether or not the pathway is a recurent one
#     :type recurrent: bool
#     :param gap: The gap between the pre and post neurons. Look below for 
#         details.
#     :type gap: int
    
#     :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
#     :return: source and target coordinate indices (in target network coordinates) 
#     :rtype: tuple   
#     """
    
#     # just for convenience
#     if profile== None:
#         w_type = None        
#         shift = {'r':0, 'phi':0}
#     else:
#         w_type = profile['type']
#         p_param = profile['params']
        
    
#     # the Watts-Strogatz p modulator
#     if np.random.rand()>p_WS:
#     	 shift['r'] = 0 # only shift a fraction p_WS of cells 
    
#     # dispalcement can be anything, but it will be rounded to the grid points
#     shift_x = shift['r'] * np.cos(shift['phi']) 
#     shift_y = shift['r'] * np.sin(shift['phi'])     
    
#     # source neuron's location on the target population
#     scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
#     s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
#     # here we draw source-centeric components of target cells, and shift them
#     # according to the desired displacement vector. If the source and target 
#     # populations are the same, i.e., connection is recurrent, and some of the
#     # drawn targets land on the source, we draw ignore them, redraw again and 
#     # again until none of the targets terminate on the source. If self-link is
#     # permitted, we don't perform this iterative step and all initially drawn
#     # (and displaced) targets will be accepted.
    
#     x = np.zeros(ncons, dtype=int)
#     y = np.zeros(ncons, dtype=int)
#     redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    
#     while sum(redraw)>0:
#         ncon = sum(redraw)
        
#         if w_type==None:
#             x_ = np.random.randint(-tcol//2, tcol-tcol//2, ncon)
#             y_ = np.random.randint(-(trow-trow//2), trow//2, ncon)
            
#         else:
#             # angle        
#             alpha = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
            
#             # radius
#             if w_type=='Gaussian':
#                 radius = p_param['std'] * np.random.randn(ncon)
#             elif w_type=='Gamma':
#                 radius = np.concatenate(
#                     (-np.random.gamma(shape= p_param['kappa'], 
#                                       scale= p_param['theta'], 
#                                       size= int(ncon // 2)),
#                     +np.random.gamma(shape=p_param['kappa'], 
#                                      scale=p_param['theta'], 
#                                      size=ncon -int(ncon // 2)))
#                     )
#             else:
#                 raise NotImplementedError
    
#             # LCRN network has a minimum gap of 1 between source and targets.
#             # If these lines don't exist, the bumps won't be pronounced. Also look
#             # at our refernce [1].
#             radius[radius<0] -= gap
#             radius[radius>=0]+= gap
            
#             # draw coordinates
#             x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
#             y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
#         # make periodic
#         # x[redraw] = x_ % tcol
#         # y[redraw] = y_ % trow
#         # make periodic around the presynapse
#         x[redraw] = (x_ + tcol/2) % tcol - tcol/2
#         y[redraw] = (y_ + trow/2) % trow - trow/2
        
        
#         # mark those self links for redraw, if necessary
#         if (recurrent) and (not self_link):
#             redraw = (x==0) & (y==0)
#         else:
#             redraw = np.zeros_like(x, dtype=bool)
            
#     # translating the coordinates w.r.t. source coordinates
#     x = (x + s_coord[0]) % tcol
#     y = (y + s_coord[1]) % trow
#     t_coords = np.array([x,y]).T 
    
#     return s_coord, t_coords.astype(int)


# def draw_post_syns_simple(s_coord, ncons,
#                   srow, scol, trow, tcol, 
#                   profile, shift={}, 
#                   self_link= False, recurrent=True, gap=1.1):
#     """Samples indices of the source neuron `s_coord` in the target network 
#     according to a location-dependent profile. The profile by default is 
#     localized on the location of the source neuron. Yet, it is possible to 
#     shift to any location relative to the source neuron on the target network. 
#     The location of the source neuron on the target network will be 
#     interpolated if the size of source and target networks don't match.

#     Periodic boundary condition is applied.

#     :param s_coord: source coordinate index as a tuple of ints
#     :type s_coord: tuple
#     :param ncons: number of out-links
#     :type ncons: int
#     :param srow: number of rows in source network
#     :type srow: int
#     :param scol: number of columns in source network
#     :type scol: int
#     :param trow: number of rows in target network
#     :type trow: int
#     :param tcol: number of columns in target network
#     :type tcol: int
#     :param profile: connectivity configuration. Please refer to 
#         :ref:`configs:Profile` for details.
#     :type profile: dict
#     :param shift: shift configuration Please refer to 
#         :ref:`configs:Anisotropy` for details.
#     :type shift: dict
#     :param self_link: Is self-link permitted; Matter only if source and target
#         populations are the same.
#     :type self_link: bool
#     :param recurrent: Whether or not the pathway is a recurent one
#     :type recurrent: bool
#     :param gap: The gap between the pre and post neurons. Look below for 
#         details.
#     :type gap: int
    
#     :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
#     :return: source and target coordinate indices (in target network coordinates) 
#     :rtype: tuple   
#     """
    
#     # just for convenience
#     p_param = profile['params']
#     w_type = profile['type']
    
#     # dispalcement can be anything, but it will be rounded to the grid points
#     shift_x = shift['r'] * np.cos(shift['phi']) 
#     shift_y = shift['r'] * np.sin(shift['phi'])     
    
#     # source neuron's location on the target population
#     scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
#     s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
#     # here we draw source-centeric components of target cells, and shift them
#     # according to the desired displacement vector. If the source and target 
#     # populations are the same, i.e., connection is recurrent, and some of the
#     # drawn targets land on the source, we draw ignore them, redraw again and 
#     # again until none of the targets terminate on the source. If self-link is
#     # permitted, we don't perform this iterative step and all initially drawn
#     # (and displaced) targets will be accepted.
#     x = np.zeros(ncons, dtype=int)
#     y = np.zeros(ncons, dtype=int)
#     redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    
#     while sum(redraw)>0:
#         ncon = sum(redraw)

#         # angle        
#         alpha = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
        
#         # radius
#         if w_type=='Gaussian':
#             radius = p_param['std'] * np.random.randn(ncon)
#         elif w_type=='Gamma':
#             radius = np.concatenate(
#                 (-np.random.gamma(shape= p_param['kappa'], 
#                                   scale= p_param['theta'], 
#                                   size= int(ncon // 2)),
#                 +np.random.gamma(shape=p_param['kappa'], 
#                                  scale=p_param['theta'], 
#                                  size=ncon -int(ncon // 2)))
#                 )
#         else:
#             raise NotImplementedError

#         # draw coordinates
#         x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
#         y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
#         # make periodic around the presynapse
#         x_ = (x_ + tcol/2) % tcol - tcol/2
#         y_ = (y_ + trow/2) % trow - trow/2
        
#         x[redraw] = x_ 
#         y[redraw] = y_
        
#         # mark those self links for redraw, if necessary
#         if (recurrent) and (not self_link):
#             dist = np.sqrt(x**2+y**2)
#             redraw = dist<=gap
#         else:
#             redraw = np.zeros_like(x, dtype=bool)
            
#     # translating the coordinates w.r.t. source coordinates
#     x = (x + s_coord[0]) % tcol
#     y = (y + s_coord[1]) % trow
#     t_coords = np.array([x,y]).T 
    
#     return s_coord, t_coords.astype(int)


def draw_posts(s_coord, ncons,
              gs_s, gs_t,
               # srow, scol, trow, tcol, 
              profile, landscape, 
              method='shift', gap=2,
              self_link= False, recurrent=True, 
              ):
    """
    Draws the coordinates of ``ncons`` postsynaptic neurons 
    according to the connectivity profile ``profile`` and the
    anisotropy pattern ``landscape`` using the given ``method``
    (``rotate`` or ``shift``). In case of recurrent pathways,
    if self-link is not allowed, autospes will be ignored and
    extra postsynapses are drawn untill ``ncons`` post that 
    differ from pre is obtained.
    """
    
   
    # source neuron's location on the target population
    dim = len(s_coord)
    scale = 1.*gs_t/gs_s
    s_coord =  np.round(s_coord*scale).astype(int)
    
    # scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
    # s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
    # initialzing containers for postsynapse coordiantes
    # x = np.zeros(ncons, dtype=int)
    # y = np.zeros(ncons, dtype=int)
    t_coords = np.zeros((ncons, dim), dtype=int)
    delays = np.zeros(ncons, dtype=float)
    
    # redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    redraw = t_coords.sum(axis=1) == 0# index of those who must be redrawn. Now, all.
    while sum(redraw)>0:
        ncon = sum(redraw)
        
        # making a (for now isotropic) point cloud around the presynapse
        if profile['type']=='homog':
            coords_ = np.random.randint(0, gs_t, size = (ncon, dim)) - s_coord
            coords_  = coords_.T # just for conveniance 
            
            # x_= np.random.randint(0,tcol, size=ncon)-s_coord[0]
            # y_= np.random.randint(0,trow, size=ncon)-s_coord[1]
        
        else:
            psi = np.random.uniform(0, 2*np.pi, ncon)
            theta = np.random.uniform(-np.pi, np.pi, ncon)
            radius = get_radial_profile(ncon, profile, gap=gap)
            x_ = radius*np.sin(psi)*np.cos(theta) 
            y_ = radius*np.sin(psi)*np.sin(theta)
            z_ = radius*np.cos(psi)
            coords_ =  np.array([x_,y_,z_])[:dim,:]
            
        # making anisotropic
        coords_ = make_anisotropic(coords_, landscape)#, method)    
        
        # make coordinates periodic around the presynapse
        
        t_coords = (coords_ + gs_t/2 ) % gs_t - gs_t/2
        delays[redraw] = np.linalg.norm(coords_, axis=0)
        # x[redraw] = (x_ + tcol/2) % tcol - tcol/2
        # y[redraw] = (y_ + trow/2) % trow - trow/2
        # delays[redraw] = np.sqrt(x_**2 + y_**2)
        
        # mark self-links for redraw, if necessary
        if (recurrent) and (not self_link):
            # redraw = (x==0) & (y==0)
            redraw = np.linalg.norm(t_coords, axis=0) == 0
        else:
            # source and target populations are different.
            # No self-link can possibly occur. 
            redraw = np.zeros_like(t_coords[0,:], dtype=bool)
            
    # translating the coordinates w.r.t. source coordinates
    t_coords = (t_coords + s_coord) % gs_t
    # t_coords = np.array([x,y]).T 
    # x = (x + s_coord[0]) % tcol
    # y = (y + s_coord[1]) % trow
    # t_coords = np.array([x,y]).T 
    
    return s_coord, t_coords.astype(int).T, delays


def get_radial_profile(nconn, profile, gap=0):
    """generate the radial profile for the inhomog. networks."""
    
    wtype = profile['type']
    param = profile['params']
    
    if wtype =='Gaussian':
        radius = param['std'] * np.random.randn(nconn)
    
    elif wtype =='Gamma':
        radius = np.concatenate(
            (-np.random.gamma(shape= param['kappa'], 
                              scale= param['theta'], 
                              size= int(nconn // 2)),
            +np.random.gamma(shape=param['kappa'], 
                             scale=param['theta'], 
                             size=nconn -int(nconn // 2)))
            )
        
    else:
        raise NotImplementedError
        
    # LCRN network can have a minimum gap of between source and targets.
    radius[radius< 0] -= gap
    radius[radius>=0] += gap
    
    return radius


def make_anisotropic(coord, lscp, method='shift'):
    """
    takes an isotropic set of coordinates and transforms them
    into anisotropic ones according to the provided ``method``.
    
    coord shape: dim*nconn
    """
    #TODO: In 3D, displacement/squeeze/... are still done in x-y plane. TO be
    #      extended.
    
    from scipy.spatial.transform import Rotation as R
    
    dim = coord.shape[0]
        
    # for convenience
    r = lscp['r']
    phi = lscp['phi']
    
    
    if method=='shift':
        if dim == 1:
            coord[0, :] += r*np.cos(phi)
        if dim == 2: 
            coord[1, :] += r*np.sin(phi)
    
        
    # elif method=='shift-rotate': # identical to shift. It's written for testing
    #     coord[:, 0] += r
    #     rot = R.from_euler('z', phi).as_matrix()[:dim,:dim]
        
    #     x, y = rot @ r0
    
    elif method=='squeeze-rotate':
        sqz = np.diag([1+r, 1/(1+r), 1])
        rot = R.from_euler('z', phi).as_matrix()
        
        # x, y = rot @ sqz @ r0 
        coord = rot[:dim,:dim] @ sqz[:dim,:dim] @ coord[:dim, :]

    elif method=='positive-rotate':
        coord[:, 0] = np.abs(coord[:, 0]) 
        rot = R.from_euler('z', phi).as_matrix()
        
        # x, y = rot @ r0
        coord = rot[:dim,:dim] @ coord[:dim, :]

    elif method=='positive-squeeze-rotate':
        coord[:, 0] = np.abs(coord[:, 0]) 
        sqz = np.diag([1+r, 1/(1+r), 1])
        rot = R.from_euler('z', phi).as_matrix()
        
        # x, y = rot @ sqz @ r0
        coord = rot[:dim,:dim] @ sqz[:dim,:dim] @ coord[:dim, :]
        
    elif method=='iso':
        pass
    
    else:
        raise
    
    # return np.round(x).astype(int), np.round(y).astype(int)
    return np.round(coord).astype(int)
    