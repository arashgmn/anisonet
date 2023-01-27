#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulations are configured via nested dictionaries. We treat the population 
and pathway configurations differently, both in the synopsis of the functions 
and their functionalities. In what follows, the structure of each configuration
is explained. In addition, one can use``Configs`` module, which provides 
ready-to-use configurations for simulations.


==========================
Population's configuration
==========================

Populations are identified by a capital character as their name (like ``"P"``) 
and must be set up as follows:


.. code-block:: python

   "P": {
         'gs': ..., 
         'noise': {'mu': ..., 'sigma': ..., 'noise_dt': ...},
         'cell': {'type': 'LIF', 
         'thr': ..., 'ref': ..., 'rest': ...,
         'tau': ..., 'C': ...}
        }


with 

#. ``gs``: grid size (int)
#. ``mu`` and ``sigma``: the background current amplitude in standard deviation as Brian quantities with current unit
#. ``noise_dt``: the time scaling of the Wiener process. Please refer to :ref:`equations:Noise scaling` for details.
#. ``type``: fixed -- for the moment only LIF neuron is possible
#. ``thr`` and ``rest``: threhold and resting potentials as Brian quantities with voltage unit
#. ``ref``: refractory period as a Brian quantity with time unit
#. ``tau``: membrane time scale as a Brian quantity with time unit
#. ``C``: membrane capacitance as a Brian quantity with capacitance unit


=======================
Pathways' configuration
=======================

Pathways are identified by two-character names (like ``"PQ"``) that specifies 
the source (P) and target (Q) populations (note the ordering). Each pathway 
must have be configured similar to the following form:

.. code-block:: python

    {'ncons': ..., 'self_link': False, 
     'profile': {...},
     'anisotropy': {...},
    }


#. ``ncons``: number of connections from each source neuron to the target population (int)
#. ``self_link``: if self-link is allowed; only important if source and target are the same object. In other words, pathway is recurrent. (bool)


~~~~~~~
Profile
~~~~~~~
``profile`` determines the `radial` profile of the connectivity which accepts a
distribution, or ``None``, i.e., no spatial structure (Erdős–Rényi network).
Some examples are provided below:
    
.. code-block:: python

    {
     ...
     
     'profile': {'type':'Gamma', 'params': {'theta': ..., 'kappa': ...} },  
     'profile': {'type':'Gaussian', 'params': {'std': ...}, gap = ... }, 
     'profile': None,
     
     ...
    }
    

#. ``type``: either ``"Gamma"`` or ``"Gaussian"`` (str)
#. ``params``: distribution parameters (float). Refer, to the documentation of 
   each distribution.
#. ``gap``: a customary float, preventing connections with distance less than
   the prescribed value. 
   
   
.. note::
    Please refer to the :refer:`anisofy:Isotropic connectivity profile` for a discussion
    on the ``gap``.


~~~~~~~
Synapse
~~~~~~~
``synapse`` are specified via the following structures:
    
.. code-block:: python

    {
     ...
     
     'synapse': {'type':'alpha_jump', 'params': {'J': ..., 'delay':..., 'tau': ...}}, # usually we use this
     'synapse': {'type':'alpha_current', 'params': {'J': ..., 'delay': ..., 'tau': ...}}, # or this
     'synapse': {'type':'alpha_conductance', 'params': {'J': ..., 'delay': ..., 'tau': ..., 'Erev': ...}}, # but not this (NOT TESTED!)
     'synapse': {'type':'tsodyks-markram_jump', 'params': {'J': ..., 'delay': ...,  'tau_f': ..., 'tau_d': ..., 'U':...}},
     ...
    }
    
    
#. ``type``: encodes both synpatic *kernel* and *model* in form of 
   ``<kernel>_<method>``. Please refer to  :ref:`equations:Synapse equations` 
   for possible values of kernels and models.
#. ``params``: 
tsodyks-markram'
   * ``tau``: synaptic timescales as a Brian time quantity (for exp, and alpha kernels)
   * ``tau_r`` and ``tau_d``: rise and decay timescales as a Brian time quantity (for biexp kernel)
   * ``delay``: synaptic delay  as a Brian time quantity
   * ``J``: synaptic qunatal with unit volt, ampere, or siemens for synapse
     models ``jump``, ``current`` or ``conductance`` respectively (c.f. :ref:`equations:Synapse equations` ).
     Note that the sign will determine the polarity of the projection (inhibitory or excitatory).
   * ``Erev``: the reversal potential for conductance-based synapse as a Brian quantity of unit volt


~~~~~~~~~~
Anisotropy
~~~~~~~~~~
The ``anisotropy`` provides a pool of parameters that can be used for enforcing
anisotropy on connectivity or synaptic properties. Whether or not such 
parameters are used in making things anisotropic, depends if the  configuration 
dictionary has ``synaptic`` and ``connectivity`` keys or not. The follwing 
examples illustrate the usage:

.. code-block::python

    {
     ...
     
     'anisotropy': None, # everything will be isotropic
     'anisotropy': {'connectivity': ..., 'params': {'r': ..., 'phi': ..., }}, # only connectivity will be anisotropic
     'anisotropy': {'synaptic': ..., 'params': {'r': ..., 'phi': ..., pmin: ..., pmax: ...}}, # only synaptic parameters will be anisotropic
     'anisotropy': {'connectivity': ..., 'synaptic': ..., 'params': {'r': ..., 'phi': ..., pmin: ..., pmax: ... }}, # both will be anisotropic
     
     ...
     }
     

Note that the parameter pool ``params`` can be used for both synaptic and 
connecitivity anistropy, depending on the method in use. c.f. :ref:`landscape` 
for details on the anisotropy methods. 

.. note::
    Ensure you understand the anisotropic methods first. They require parameters
    that must be provided in the ``params`` pool. If not given, an error will 
    be raised. 
    
The entries in the parameters pool depend on the anisotropy method in use. c.f.
:ref:`landscape` for information on valid forms of configuring them.
    
    

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
"""


from brian2.units import pA, mV, ms, pF, nA
import numpy as np
from copy import deepcopy
from pdb import set_trace

np.random.seed(18)

# VALID ARGUMETNS
VALID_CELL_TYPES=['LIF']
VALID_SYN_TYPES=['conductance', 'current', 'jump']
VALID_SYN_KERNELS = ['jump','alpha','exp','tsodyks-markram']
VALID_PROFILES = ['Gamma', 'Gaussian']
VALID_NONUNIFORMITY = ['connectivity','tau_f', 'tau_d', 'U', 'J']
VALID_SYN_METHODS = ['cos', 'sin', 'tan', 'normal']
VALID_CON_METHODS = ['shift', 'shift-rotate', 'squeeze-rotate', 
                     'positive-rotate','positive-squeeze-rotate']

# MANDATORY ARGS
MAN_ARGS_NOISE = set(['mu', 'sigma', 'noise_dt'])
MAN_ARGS_CELL = set(['thr', 'rest', 'tau', 'ref', 'C'])
MAN_ARGS_PROFILE = {
    'Gamma': set(['theta','kappa']),
    'Gaussian': set(['std', 'gap'])
    }
MAN_ARGS_SYN_KERNEL = {
    'alpha': set(['tau']),
    'biexp': set(['tau_d', 'tau_r']),
    'exp'  : set(['tau']),
    'tsodyks-markram':  set(['tau_d', 'tau_f', 'U']),
    'STDP': set(['taupre','taupost'])
    }
MAN_ARGS_SYN_TYPE = {
    'conductance': set(['J', 'delay' ,'Erev']),
    'current': set(['J', 'delay']),
    'jump': set(['J', 'delay'])
    }
MAN_ARGS_LANDSCAPE= {
    'random': set(['vmin','vmax']),
    'perlin': set(['vmin','vmax', 'scale'])
    }

MAN_ARGS_NONUNIF_METHODS = {
    'connectivity': {valid: set(['r', 'phi']) for valid in VALID_CON_METHODS},
    'U': {valid: set(['phi','Umin','Umax']) for valid in VALID_SYN_METHODS},
    'tau_f': {valid: set(['phi','tau_fmin','tau_fmax']) for valid in VALID_SYN_METHODS},
    'tau_d': {valid: set(['phi','tau_dmin','tau_dmax']) for valid in VALID_SYN_METHODS},
     }
for variable in ['U','tau_f','tau_d']:
    MAN_ARGS_NONUNIF_METHODS[variable]['normal'].add('s'+variable)



def config_checker(config_obj, autoremove=True):
    
    def check_mandatories(mandatory_set, set_, name):
        msg = f'The following are mandatory args for {name} but are not given:\n{mandatory_set-set_}'
        assert len(mandatory_set-set_)==0, msg
            
    def check_extra(mandatory_set, set_, name):
        if len(set_-mandatory_set)>0:
            for kw in set_-mandatory_set:
                print(f'INFO: {kw} is not mandatory for {name}.')
    
    
    ## CHECKING POPS ##            
    for pop, pop_cfg in config_obj.pops_cfg.items():
        assert type(pop_cfg['gs'])==int
        
        # noise
        args_noise = set(pop_cfg['noise'].keys())
        check_mandatories(MAN_ARGS_NOISE, args_noise, 'noise')
        check_extra(MAN_ARGS_NOISE, args_noise, 'noise')
        
        # cell
        args_cell = set(pop_cfg['cell']['params'].keys())
        check_mandatories(MAN_ARGS_CELL, args_cell, 'cell')
        check_extra(MAN_ARGS_CELL, args_cell, 'cell')
        
    
    ## CHECKING PATHWAY NAMES ##            
    pop_list = config_obj.pops_cfg.keys()
    for pathway in config_obj.conns_cfg.keys():
        src, trg = pathway
        assert src in pop_list, f'source {src} is not a population in pathway {pathway}'
        assert trg in pop_list, f'target {trg} is not a population in pathway {pathway}'
            
            
    ## CHECKING NONUNIFORMITY & LANDSCAPE ##            
    for obj, nonunif in config_obj.nonuniformity.items():
        assert obj in config_obj.lscps_cfg, f'A nonuniformity is defined for {obj} but no landscape parameters'
        
        for on, method in nonunif.items():
            #set_trace()
            man_args_nonunif_method = set(MAN_ARGS_NONUNIF_METHODS[on][method])
            args_nonunif_method = set(config_obj.lscps_cfg[obj].keys())
            
            name = f'nonuniformity on {on} with method {method}'
            check_mandatories(man_args_nonunif_method, args_nonunif_method, name)
            check_extra(man_args_nonunif_method, args_nonunif_method, name)
            
        for lscp_name, cfg in config_obj.lscps_cfg[obj].items():
            if type(cfg)==dict:
                man_args_lscp = MAN_ARGS_LANDSCAPE[cfg['type']]
                args_lscp = set(cfg['args'])
                name = f'landscape {lscp_name} with type {cfg["type"]}'
                check_mandatories(man_args_lscp, args_lscp, name)
                check_extra(man_args_lscp, args_lscp, name)
                
            
    ## CHECKING PROFILE & ANISOTROPY & SYNAPSES ##            
    for pathway, conn_cfg in config_obj.conns_cfg.items():
        assert type(conn_cfg['ncons'])==int
        assert type(conn_cfg['self_link'])==bool
            
    
        ## SYNAPSE CONFIGS ##
        syn_cfg = conn_cfg['synapse']
        syn_kern, syn_type = syn_cfg['type'].split('_')

        man_args_syn_kern = MAN_ARGS_SYN_KERNEL[syn_kern]
        man_args_syn_type = MAN_ARGS_SYN_TYPE[syn_type]
        
        man_args_syn = man_args_syn_type.union(man_args_syn_kern)
        args_syn = set(syn_cfg['params'].keys())
        check_mandatories(man_args_syn, args_syn,'synapse')
        check_extra(man_args_syn, args_syn,'synapse')
        
        ## CHECKING PROFILE ##            
        if 'profile' in conn_cfg: 
            prof_cfg = conn_cfg['profile']
            
            assert 'type' in prof_cfg
            assert 'params' in prof_cfg
            
            man_args_prof = MAN_ARGS_PROFILE[prof_cfg['type']]
            args_prof = set(prof_cfg['params'].keys())
            check_mandatories(man_args_prof, args_prof, 'profile')
            check_extra(man_args_prof, args_prof, 'profile')

            
    ## CHECKING STIM ## 
    for stim_name, cfg in config_obj.stims_cfg.items():
        #set_trace()
        pop, id_ = stim_name.split('_') 
        assert pop in pop_list, f'Unrecognized population to stimulate with {pop}'
        assert id_.isdigit(), 'Stimulation name mus comply with the format `<pop_name>_<digit>`.'
        


def round_to_even(gs, scaler):
    """
    We better round things to even number for better visualization
    """
    rounded = round(gs/scaler)
    if rounded%2:
        rounded+=1
    return int(rounded)


def make_config(name='EI_net', scalar=3):
    """
    Generates the population and pathways config dictuinary only by providing 
    the name of the desired network. 
    
    .. note::
        One should differetiate between homogeneity/randomness in angle and 
        location. `[1]`_ used these terms somewhat loosely. We use the following 
        terms for different setups:
            
            * ``homiso_net``: Homogenous and isotropic netowrk, equivalent to the fully
              random graph of Erdos-Renyi.
            * ``iso_net``: Isotropic but spatially inhomogeneous (in a locally 
              conneted manner, although with a long-tailed radial profile one can
              generate few long-range connection -- thus produce a small-world net).
            * ``homo_net``: Connections are formed without dependence on the distance,
              but angle.
            * ``I_net``: the recurrent inhibitory network with radial and angular
              profiles according to to `[1]`_.
            * ``EI_net``: the recurrent inhibitory network with radial and angular
              profiles according to to `[1]`_.

        Also note that these structures are independent from how anisotropy is 
        imposed.

    .. note::
        It is possible to decrease the network's grid size by a factor of 
        ``scalar``. However, such shrinkage has different effects on different
        networks. One uni-population networks, the synaptic strenght is 
        enlarged by a factor of ``scalar**2`` to account for lower number of 
        afferents. However, syanptic weights are left intact for the 
        two-population networks, since they are set up in balance and afferents
        will effectively cancel each other. An exception from this rule is the
        signle-population excitatory network. This network is inherently 
        unstable. So, we did not enlarged the synaptic weights partially to 
        avoid blow-up. In other words, the synaptic weights is large enough to
        trigger spike but not large enough to propagate it too far.


    :param name: nework name
    :type name: str, optional
    :param scalar: scales down the network by a factor. The network must be 
        divisble to the factor. Number of connections, their strenghts, and the
        connectivity profile will be scaled accordingly. defaults to 3
    :type scalar: int, optional
    :return: pops_cfg, conn_cfg 
    :rtype: tuple of dicts

    .. _[1]: https://doi.org/10.1371/journal.pcbi.1007432

    """
    if name=='test':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {
                       'synaptic': 'cos',
                       'connectivity': 'shift', 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'perlin', 'args': {'scale':3} }, 
                                  'U': {'type': 'perlin', 'args': {'scale':2}, 'vmin': 0.01, 'vmax':0.3},
                                  'Umin':0.1, 'Umax':0.4
                                  }  
                       },
                   
                'training': {'type': 'STDP', 
                             'params': {'taupre': 10*ms,'taupost': 10*ms,
                                        'Apre': 0.05, 'Apost': -0.055,},
                             },
                }
            }
        
        stim_cfg = {}
        
    elif name=='demo':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'connectivity': 'shift', 
                                  'params': {'r'  : 1, 
                                             'phi': {'type': 'perlin', 'args': {'scale':2} },
                                             }  
                                  },
                   
                   }
        }
        
        stim_cfg = {}
        
    elif name=='I_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   
                   # Note: In the paper, four different anisotropic configurations 
                   # are introduced: "random", "symmetric", "homogeneous" and 
                   # "perlin". To emulate these cases follow these instructions:
                       # comment the entire `anisotropy` dict (as well as the key) to get "random"
                       # use `'phi': {'type': 'random'}`` to get "symmetric"
                       # use `'phi': np.pi/6`` to get "homogeneous"
                       # use `'phi': {'type': 'perlin', ...` to get "perlin"
                   'anisotropy': {
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'perlin', 'args': {'scale':3} }, # a perlin-generate angular landscape
                                  #'phi': {'type': 'random'},     # a random angular ladscape
                                  #'phi': np.pi/6,  # a homogeneous angular ladscape
                                  }  
                       },
                   }
        }
        
        stim_cfg = {}
        
    elif name=='E_net':
        pops_cfg = {
            'E': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 50*pA, 'sigma': 400*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  }
            }

        conn_cfg = {
            'EE': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': 2.5*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {
                        #'synaptic': 'cos',       # induces anisotropy in synapses with cosine method
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'perlin', 'args': {'scale':4} },
                                  'U': {'type': 'perlin', 'args': {'scale':2}, 'vmin': 0.01, 'vmax':0.3},
                                 }  
                       },
                   },
        }
    
        stim_cfg = {}
        
        
    elif name=='EI_net':
        pops_cfg = {
            'I': {'gs': round_to_even(60, scalar), 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  },
            
            'E': {'gs': round_to_even(120, scalar), 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  },
        }
        
        conn_cfg = {
            'EE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                  'profile': {'type':'Gaussian', 'params': {'std': 9/scalar}, 'gap': max(2, 6./scalar) },
                  'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms} },
                  
                  # Note: In the paper, four different anisotropic configurations 
                  # are introduced: "random", "symmetric", "homogeneous" and 
                  # "perlin". To emulate these cases follow these instructions:
                      # comment the entire `anisotropy` dict (as well as the key) to get "random"
                      # use `'phi': {'type': 'random'}`` to get "symmetric"
                      # use `'phi': np.pi/6`` to get "homogeneous"
                      # use `'phi': {'type': 'perlin', ...` to get "perlin"
                  'anisotropy': {
                      'connectivity': 'shift', # induces anisotropy in connections with shift method 
                      'params': {'r'  : 1, 
                                 'phi': {'type': 'perlin', 'args': {'scale':2} }, # a perlin-generate angular landscape
                                 #'phi': {'type': 'random'},  # a random angular ladscape
                                 #'phi': np.pi/6,  # a homogeneous angular ladscape
                                 }  
                      },
                  },
                  
            'EI': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 4.5/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'random'},  # a random angular ladscape
                                  }  
                       },
                   },
            
            'IE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 12/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'random'},  # a random angular ladscape
                                  }  
                       },
                   },

            'II': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 6/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'random'},  # a random angular ladscape
                                  }  
                       },
                   },
        }
        
        stim_cfg = {}
        
    elif name=='homo_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                               'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                               'tau': 10*ms, 'C': 250*pF
                               }
                          }
                  }
            }
        
        # Note: For a homogeneous network, `profile` entry can be omitted. 
        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'perlin', 'args': {'scale':3} },
                                  }  
                       },
                   },
        }

        stim_cfg = {}
                
    elif name=='iso_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                               'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                               'tau': 10*ms, 'C': 250*pF
                               }
                          }
                  }
            }
        
        # Note: We can model an isotropic network, simply by omitting the 
        #       anisotropy key in the connections config
        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 5./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   },
        }    

        stim_cfg = {}
        

    elif name=='homiso_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                               'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                               'tau': 10*ms, 'C': 250*pF
                               }
                          }
                  }
            }
        
        # Note: For a homogeneous and isotropic network no `profile` or
        #       `anisotropy` is needed.
        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   },
        }    
    
        stim_cfg = {}
        
    elif name=='STSP_TM_I_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                               'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                               'tau': 10*ms, 'C': 250*pF
                               }
                           }
                  }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 5./scalar) },
                   'synapse': {'type':'tsodyks-markram_jump', 
                               'params': {'J': -0.221*mV*(scalar**2), 'delay':1*ms, 
                                          'tau': 10*ms, 'tau_f': 1500.*ms, 'tau_d': 200.*ms, 
                                          }},
                   'anisotropy': {
                       'connectivity': 'shift', 'synaptic': 'cos',
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'perlin', 'args': {'scale':2} },
                                  # TODO: should somehow detect if U is a landscape or will be drawn from Umin to Umax.
                                  'U': {'type': 'perlin', 'args': {'scale':2}, 'vmin': 0.01, 'vmax':0.3},
                                  'Umin': 0.05, 'Umax':0.3
                                  }  
                       },
                   },
        }
        
        stim_cfg = {}
           
    elif name=='STSP_TM_EI_net':
        pops_cfg = {
            'I': {'gs': round_to_even(60, scalar), 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  },
            
            'E': {'gs': round_to_even(120, scalar), 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'params': {
                                'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                                'tau': 10*ms, 'C': 250*pF
                                }
                           }
                  }
            }


        conn_cfg = {
            'EE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                  'profile': {'type':'Gaussian', 'params': {'std': 9/scalar}, 'gap': max(2, 6./scalar) },
                  'synapse': {'type':'tsodyks-markram_jump', 
                              'params': {'J': 0.221*mV*(scalar**2), 'delay':1*ms, 
                                         'tau_f': 1500*ms, 'tau_d': 600*ms, 'U':0.1}},
                  'anisotropy': {
                      'connectivity': 'shift', 
                      'params': {'r'  : 1, 'phi': {'type': 'perlin', 'args': {'scale':2} }, }  
                      },
                  },
            
            'EI': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 4.5/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'tsodyks-markram_jump', 
                               'params': {'J': 0.221*mV*(scalar**2), 'delay':1*ms, 
                                          'tau_f': 1500*ms, 'tau_d': 600*ms, 'U':0.1}},
                   
                   },
            
            'IE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 12/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   },

            'II': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 6/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   },
            
            }
    
        stim_cfg = {}
        
    
    elif name=='EI_net_focal_stim':
        pops_cfg = {
            'I': {'gs': round_to_even(60, scalar), 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  },
            
            'E': {'gs': round_to_even(120, scalar), 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                 }
            }


        conn_cfg = {
            'EE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                  'profile': {'type':'Gaussian', 'params': {'std': 9/scalar}, 'gap': max(2, 6./scalar) },
                  'synapse': {'type':'tsodyks-markram_jump', 
                              'params': {'J': 0.221*mV*(scalar**2), 'delay':1*ms, 
                                         'tau_f': 1500*ms, 'tau_d': 600*ms, 'U':0.1}},
                  'anisotropy': {
                      'connectivity': 'shift', 
                      'params': {'r'  : 1, 'phi': {'type': 'perlin', 'args': {'scale':2} }, }  
                      },
                  },
            
            'EI': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 4.5/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'tsodyks-markram_jump', 
                               'params': {'J': 0.221*mV*(scalar**2), 'delay':1*ms, 
                                          'tau_f': 1500*ms, 'tau_d': 600*ms, 'U':0.2}},
                   
                   },
            
            'IE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 12/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   },

            'II': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 6/scalar}, 'gap': max(2, 6./scalar) },
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   },
            
            }
            
        stim_cfg = {
            'E_0': {'type': 'const', 'I_stim': 500*pA,
                    'domain': {'type': 'r', 'x0': 15, 'y0': 20, 'r':2}
                    },
            
            # 'I_1': {'type': 'const', 'I_stim': -0*pA,
            #         'domain': {'type': 'random', 'p': 1}
            #         }
            
            }
            
    elif name=='I_net_focal_stim':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }
        
        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}, 'gap': max(2, 6./scalar) },
                   
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   # 'synapse': {'type':'tsodyks-markram_jump', 
                   #             'params': {'J': 0.221*mV*(scalar**2), 'delay':1*ms, 
                   #                        'tau_f': 1500*ms, 'tau_d': 600*ms, 'U':0.1}},
                   
                   
                   'anisotropy': {
                       'connectivity': 'shift', # induces anisotropy in connections with shift method 
                       'params': {'r'  : 1, 
                                  'phi': {'type': 'perlin', 'args': {'scale':4} }, # a perlin-generate angular landscape
                                  #'phi': {'type': 'random'},  # a random angular ladscape
                                  #'phi': np.pi/6,  # a homogeneous angular ladscape
                                  }  
                       },
                   
                   'training': {'type': 'STDP', 
                             'params': {'taupre': 10*ms,'taupost': 10*ms,
                                        'Apre': 0.05, 'Apost': -0.055,},
                             },
                
                   }
        }
           
        stim_cfg = {
            'I_0': {'type': 'const', 'I_stim': 400*pA,
                    'domain': {'type': 'r', 'x0': 20, 'y0': 10, 'r':2.5}
                    },
            
            # 'I_1': {'type': 'const', 'I_stim': 700*pA,
            #         'domain': {'type': 'r', 'x0': 3, 'y0': 10, 'r': 7}
            #         }
            
            }
   
    else:
        raise
    
    return pops_cfg, conn_cfg, stim_cfg
       

    
class Configer(object):
    
    def __init__(self, scale=1,
                 pops_cfg = {}, 
                 conns_cfg = {}, 
                 nonuniformity = {},
                 lscps_cfg = {},
                 stims_cfg = {},
                 ):
        """
        Provides an interface to generate the default configurations values. 
        With some methods, it supports modifying these values to the desired 
        ones.
        """
        
        self.scale = scale
        self.pops_cfg = pops_cfg
        self.conns_cfg = conns_cfg 
        self.nonuniformity= nonuniformity
        self.lscps_cfg = lscps_cfg 
        self.stims_cfg = stims_cfg
        
    
    def add_pop(self, name, gs):
        pop_cfg = {}
        pop_cfg['gs'] = gs
        pop_cfg['noise'] = self.make_noise()
        pop_cfg['cell'] = self.make_cell()
        
        if name in self.pops_cfg:
            self.pops_cfg[name].update(pop_cfg)
        else:
            self.pops_cfg[name] = pop_cfg
        del pop_cfg
        
        
    def add_pathway(self, src, trg, ncons, self_link=False):
        """adds the defauls"""
        conn_cfg = {}
        conn_cfg['ncons'] = ncons
        conn_cfg['self_link'] = self_link
        conn_cfg['synapse'] = self.make_syn()
        #conn_cfg['profile'] = self.make_profile()
        
        
        if src+trg in self.pops_cfg:
            self.pops_cfg[src+trg].update(conn_cfg)
        else:
            self.conns_cfg[src+trg] = conn_cfg
        
        del conn_cfg
        
    def add_profile(self, pathway, prof_type, **params):
        assert pathway in self.conns_cfg.keys(), f'Pathway {pathway} not recognized.'
        self.conns_cfg[pathway]['profile'] = self.make_profile(prof_type, **params)
   
        
    def add_nonuniformity(self, obj, on='connectivity', method='shift',):
        assert on in VALID_NONUNIFORMITY, 'Non-uniformity can be applied only on {}'.format(VALID_NONUNIFORMITY)
        
        if on =='connectivity':
            assert method in VALID_CON_METHODS, 'Only the following anisotropic connectivity methods are supported: {}'.format(VALID_CON_METHODS)
        else: # on=='synaptic':
            assert method in VALID_SYN_METHODS, 'Only the following anisotropic synaptic methods are supported: {}'.format(*VALID_SYN_METHODS)
            
        nonuniformity = {on: method}
        if obj in self.nonuniformity:
            self.nonuniformity[obj].update(nonuniformity)
        else:
            self.nonuniformity[obj] = nonuniformity
        
        del nonuniformity
            
    def add_landscape(self, obj, param_name, **params):
        
        lscp = {param_name: self.make_landscape(**params)}
        if obj in self.lscps_cfg:
            self.lscps_cfg[obj].update(lscp)
        else:
            self.lscps_cfg[obj] = lscp
        
        del lscp
    
    def add_stim(self, name, I_stim, **params):
        self.stims_cfg.update( self.make_stim(name, I_stim, **params) )
        
    # TODO
    def add_training(self):
        pass
        
    
    #### CONFIG GENERATORS 
    def make_noise(self, **params):
        cfg = {'mu': 0*pA, 'sigma': 0*pA, 'noise_dt': 1*ms}
        if params!= None:
            cfg.update(params)
        return cfg
    
    
    def make_cell(self, cell_type='LIF', **params):
        assert cell_type in VALID_CELL_TYPES, 'Only {} neuron types are supported.'.format(VALID_CELL_TYPES)
        
        cfg = {'type': 'LIF', 
              'params': dict(thr=-55*mV, rest=-70*mV, 
                            tau=10*ms, ref=2*ms, 
                            C = 250*pF)
              }
        if params!=None:
            cfg['params'].update(params)
        
        return cfg
        
    
    def make_syn(self, syn_type='current', syn_kern='alpha', **params):
        assert syn_type in VALID_SYN_TYPES, 'Only {} synapse types are supported'.format(VALID_SYN_TYPES)
        assert syn_kern in VALID_SYN_KERNELS, 'Only {} synapse kernels are supported'.format(VALID_SYN_KERNELS)
    
        cfg = {'type': syn_kern+'_'+syn_type,
               'params': dict(delay=1*ms)
               }
        
        if syn_type =='jump': 
            cfg['params']['J'] = -0.221*mV*(self.scale**2)
        elif syn_type =='current':
            cfg['params']['J'] = -10*pA*(self.scale**2)
        else:
            pass # must be adapted for conductances
        
        if syn_kern == 'tsodyks-markram':
            cfg['params']['tau_f'] = 1000*ms
            cfg['params']['tau_d'] = 200*ms
            cfg['params']['U'] = 0.2
        else:
            cfg['params']['tau'] = 5*ms 
            
        if params!=None:
            cfg['params'].update(params)
        
        return cfg
        
    
    def make_profile(self, prof_type, 
                     theta=None, kappa=None,
                     std=None, gap=None):
        assert prof_type in VALID_PROFILES, 'Only {} connectivity profiles are supported'.format(VALID_PROFILES)
        
        cfg = {'type': prof_type}
        
        if prof_type=='Gaussian':
            assert std!=None
            params = dict(std=std)
            
        else:
            assert kappa!=None and theta!=None
            params = dict(kappa=kappa, theta=theta)
            
        if gap==None:
            gap = max(2, 6./self.scale)
        
        params['gap'] = gap
        cfg['params'] = params
        
        return cfg
        
        
    def make_landscape(self, val=None, vmin=None, vmax=None, vals=None,
                      mode='perlin', scale=3):
        if val!=None:
            assert type(val)==int or type(val)==float
            return val
        
        if mode == 'constants':
            assert vals != None and type(vals)==dict
            return {'type': 'constant', 'args': vals}
            
        else:
            assert vmin!=None and vmax!=None
            
            if mode == 'perlin':
                return {'type': 'perlin', 
                        'args': {'vmin':vmin, 'vmax':vmax, 'scale':scale}
                        }
            elif mode == 'random':
                return {'type': 'random', 
                        'args': {'vmin':vmin, 'vmax':vmax}
                        }
            
            
     
    def make_stim(self, name, I_stim, 
                  x0=None, y0=None, r=None, 
                  x_min=None, x_max=None, y_min=None, y_max=None,
                  domain_type='r'):
        
        cfg = {}
        cfg[name] = {'type': 'const', 
                     'I_stim': I_stim, 
                     }
        domain = {'type': domain_type}
        if domain_type == 'xy':
            assert x_min!=None and x_max!=None and y_min!=None and y_max!=None
            
            domain['x_min'] = x_min
            domain['x_max'] = x_max
            domain['y_min'] = y_min
            domain['y_max'] = y_max
        else:
            assert x0!=None and y0!=None and r!=None
            
            domain['x0'] = x0
            domain['y0'] = y0
            domain['r'] = r
        
        cfg[name]['domain'] = domain
        return cfg
        
        
        
    # def make_landscape(self, on, method, **params):
        
        # if on == 'connectivity':
        #     # all connectivity methods need the following two
        #     cfg = {'r': 1, 'phi': np.pi/3}
        
        # else: # on=='synaptic'
        #     cfg = {'U': (0.1, 0.3) }
            
        # if params!=None:
        #     cfg.update(params)
        
        # return cfg
        
   
         
    #### CONFIG UPDATORS 
    def update_noise(self, name, dict_, reset=False):
        if not reset:
            self.pops_cfg[name]['noise'].update(dict_)
        else:
            self.pops_cfg[name]['noise'] = dict_
            
            
    def update_cell_params(self, name, dict_, reset=False):
        if not reset:
            self.pops_cfg[name]['cell']['params'].update(dict_)
        else:
            self.pops_cfg[name]['cell']['params'] = dict_
                
        
    def update_syn(self, pathway, syn_kern=None, syn_type=None, **params):
        syn_kern_, syn_type_ = self.conns_cfg[pathway]['synapse']['type'].split('_')
        if syn_kern != None:
            assert syn_kern in VALID_SYN_KERNELS, f'invalid synaptic kernel: {syn_kern}'
            syn_kern_ = syn_kern
        if syn_type != None:
            assert syn_type in VALID_SYN_TYPES, f'invalid synaptic type: {syn_type}'
            syn_type_ = syn_type
        
        # self.conns_cfg[pathway]['synapse']['type'] = syn_kern_ +'_'+ syn_type_ 
        self.conns_cfg[pathway]['synapse'] = self.make_syn(syn_type_,syn_kern_)    
        
        if params!=None:
            self.conns_cfg[pathway]['synapse']['params'].update(params)
            
    def update_syn_params(self, pathway, dict_, reset=False):
        if not reset:
            self.conns_cfg[pathway]['synapse']['params'].update(dict_)
        else:
            self.conns_cfg[pathway]['synapse']['params'] = dict_
        
        
    
    def update_profile_type(self, pathway, new_type):
        assert new_type in VALID_PROFILES, f'invalid connectivity profile type: {new_type}'
        self.conns_cfg[pathway]['profile']['type'] = new_type
        
    def update_profile_params(self, pathway, dict_, reset=False):
        if not reset:
            self.conns_cfg[pathway]['profile']['params'].update(dict_)
        else:
            self.conns_cfg[pathway]['profile']['params'] = dict_
        
        
        
    def update_nonuniformity(self, obj, on, method, reset=False):
        assert on in VALID_NONUNIFORMITY, f'invalid nonuniformity: {on}'
        assert method in (VALID_CON_METHODS+VALID_SYN_METHODS), f'invalid nonuniformity method: {method}'
        
        if not reset:
            self.nonuniformity[obj].update({on:method})
        else:
            self.nonuniformity[obj] = {on:method}
        
        
    def update_landscape(self, pathway, dict_, reset=False):
        if not reset:
            self.lscps_cfg[pathway].update(dict_)
        else:
            self.lscps_cfg[pathway] = dict_
        
        
    def update_stimulation(self, stim_name, dict_, reset=False):
        if not reset:
            self.stim_cfg[stim_name].update(dict_)
        else:
            self.stim_cfg[stim_name] = dict_
        
        
    def get(self):
        config_checker(self)
        return self.aggregate()
        
    
    def aggregate(self):
        return (self.pops_cfg, 
                self.conns_cfg,
                self.nonuniformity,
                self.lscps_cfg,
                self.stims_cfg, 
                )
    

# def make_aniso_config(mode, val=None, vmin=None, vmax=None, scale=3):
#     if mode=='const':
#         assert val!=None
#         return val
#     elif mode=='range':
#         assert vmin!=None and vmax!=None
#         return (vmin, vmax)
#     elif mode=='perlin':
#         assert vmin!=None and vmax!=None
#         return {'type': 'perlin' ,'args':{'scale':scale}}
#     elif mode=='random':
#         return {'type': 'random'}
#     else:
#         raise NotImplementedError('Anisotropic mode not recognized.')


    
def get_config(name, scale):
    print(f'I am searching for {name}')
    if name == 'homiso_net':
        c = deepcopy(Configer(scale))
        c.add_pop('I', round_to_even(100, scale))
        c.update_noise('I', {'mu': 700*pA, 'sigma': 100*pA})
        c.add_pathway('I', 'I', ncons =round_to_even(1000, scale**2))
        
    elif name == 'iso_net':
        c = deepcopy(get_config('homiso_net', scale))
        c.add_profile('II', 'Gamma', theta=3/c.scale, kappa= 4)
        
    elif name == 'homo_net':
        c = deepcopy(get_config('homiso_net', scale))
        c.add_nonuniformity('II', on='connectivity', method='shift')
        c.add_landscape('II', 'r', val=1)
        c.add_landscape('II', 'phi', vmin=0, vmax=2*np.pi, mode='perlin')
        
    elif name =='I_net':
        c = deepcopy(get_config('homiso_net', scale))
        c.add_profile('II', 'Gamma', theta=3/c.scale, kappa= 4)
        c.add_nonuniformity('II', on='connectivity', method='shift')
        c.add_landscape('II', 'r', val=1)
        c.add_landscape('II', 'phi', vmin=0, vmax=2*np.pi, mode='perlin')
    
    elif name == 'E_net':
        c = deepcopy(Configer(scale))
        
        c.add_pop('E', round_to_even(100, scale))
        c.update_noise('E', {'mu': 50*pA, 'sigma': 400*pA})
        
        c.add_pathway('E', 'E', ncons =round_to_even(1000, scale**2))
        c.update_syn_params(pathway='EE', dict_={'J':2.5*(scale**2)*pA})
    
        c.add_profile('EE', 'Gamma', theta=3/c.scale, kappa= 4)
        c.add_nonuniformity('EE', on='connectivity', method='shift')
        c.add_landscape('EE', 'r', val=1)
        c.add_landscape('EE', 'phi', vmin=0, vmax=2*np.pi, mode='perlin')
    
    elif name == 'EI_net':
        c = deepcopy(Configer(scale))
        
        c.add_pop('I', round_to_even(60, scale))
        c.add_pop('E', round_to_even(120, scale))
        c.update_noise('I', {'mu': 350*pA, 'sigma': 100*pA})
        c.update_noise('E', {'mu': 350*pA, 'sigma': 100*pA})
        
        c.add_pathway('E','E', ncons=round_to_even(720, scale**2))
        c.add_pathway('E','I', ncons=round_to_even(180, scale**2))
        c.add_pathway('I','E', ncons=round_to_even(720, scale**2))
        c.add_pathway('I','I', ncons=round_to_even(180, scale**2))

        J = 10*(scale**2)*pA
        g = 8
        c.update_syn_params(pathway='EE', dict_ = {'J': J})
        c.update_syn_params(pathway='EI', dict_ = {'J': J})
        c.update_syn_params(pathway='IE', dict_ = {'J': -g*J})
        c.update_syn_params(pathway='II', dict_ = {'J': -g*J})
        
        
        c.add_profile(pathway='EE', prof_type='Gaussian', std=9/scale)
        c.add_profile(pathway='EI', prof_type='Gaussian', std=4.5/scale)
        c.add_profile(pathway='IE', prof_type='Gaussian', std=12/scale)
        c.add_profile(pathway='II', prof_type='Gaussian', std=6/scale)
        
        c.add_nonuniformity(obj='EE', on='connectivity', method='shift')
        c.add_landscape(obj='EE', param_name='r', val=1)
        c.add_landscape(obj='EE', param_name='phi', 
                        vmin=0, vmax=2*np.pi, mode='perlin')
    
    
    elif name == 'I_net_syn_TM':
        c = deepcopy(get_config('I_net', scale))
        c.update_syn('II', syn_kern='tsodyks-markram', syn_type='jump')
    
    elif name == 'EI_net_syn_TM':
        c = deepcopy(get_config('EI_net', scale))
        
        J = 0.221*mV*scale**2
        g = 8
        c.update_syn('EE', syn_kern='tsodyks-markram', syn_type='jump', J=J)
        c.update_syn('EI', syn_kern='tsodyks-markram', syn_type='jump', J=J)
        c.update_syn('IE', syn_kern='tsodyks-markram', syn_type='jump', J=-g*J)
        c.update_syn('II', syn_kern='tsodyks-markram', syn_type='jump', J=-g*J)
        
        
    elif name == 'I_net_stim':
        c = deepcopy(get_config('I_net', scale))
        c.add_stim('I_0', I_stim=500*pA, x0=20, y0=10, r=4)
    
    
    elif name == 'EI_net_stim':
        c = deepcopy(get_config('EI_net', scale))
        c.add_stim('E_0', I_stim=500*pA, x0=20, y0=10, r=2.5)
        
    elif name == 'I_net_synaniso':
        c = deepcopy(get_config('I_net_syn_TM', scale))
        c.add_nonuniformity('II', on='U', method='normal')
        c.add_landscape('II', param_name='Umin', val=0.1)
        c.add_landscape('II', param_name='Umax', val=0.3)
        c.add_landscape('II', param_name='s', val=1)
    
    elif name == 'I_net_syn_TM_synaniso':
        c = deepcopy(get_config('I_net_syn_TM', scale))
        c.add_nonuniformity('II', on='tau_f', method='cos')
        c.add_landscape('II', param_name='tau_fmin', val=0.8)
        c.add_landscape('II', param_name='tau_fmax', val=2.5)
        
        c.add_nonuniformity('II', on='tau_d', method='normal')
        c.add_landscape('II', param_name='tau_dmin', val=0.1)
        c.add_landscape('II', param_name='tau_dmax', val=0.3)
        c.add_landscape('II', param_name='s', val=1)

    else:
        raise
        
    return c
        

# class Foo(object):
#     def __init__(self,):
#         self.my_char = ''
        
#     def add_char(self, new_char=''):
#         self.my_char  = self.my_char + new_char
        

# def debug(char):
    
#     if char=='a':
#         x = Foo()
#         x.add_char('a')
#     elif char =='ab':
#         x = debug('a')
#         x.add_char('b')
#     elif char =='abc':
#         x = debug('ab')
#         x.add_char('c')
#     else:
#         raise
#     return x
        
#     if name =='a':
#         result = name
#     elif name =='ab':
#         result = debug('a') + 'b'
#     elif name =='abc':
#         result = debug('ab') + 'c'
#     else:
#         result = ''
#     return result

# import gc
if __name__ == '__main__':
    
    a = get_config('I_net_syn_TM',scale=4)
    print(a, id(a))
    #del c
    # gc.collect()
    
    b = get_config('EI_net_syn_TM',scale=3)
    print(b, id(b))
    #del c
    # gc.collect()

    d = get_config('I_net_syn_TM',scale=2)
    print(d, id(d))
    
    # a = get_config('iso_net',scale=1)
    # print(a, id(a))
    
    # b = get_config('homiso_net',scale=3)
    # print(b, id(b))
    