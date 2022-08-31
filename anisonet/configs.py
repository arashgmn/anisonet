#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configs is a utility module that passes the default configurations according to
the `[1]`_. Configurations are set-up in form of nested dictionaries whose 
structure is as follows:
    
.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
"""

from brian2.units import pA, mV, ms, pF
import numpy as np

np.random.seed(8)

# pops_cfg0 = {
#     'E': {'gs':10*r, 'mu': 700*pA, 'sigma': 250*pA},
#     'I': {'gs':5*r, 'mu': 100*pA, 'sigma': 50*pA}
# }


# syns_cfg0 = { 
#     'EE': {'synapse_dyn': 'alpha', 'ncons':10*r, 'J': 0.22*mV},
#     'EI': {'synapse_dyn': 'alpha', 'ncons':10*r, 'J': 0.22*mV},
#     'IE': {'synapse_dyn': 'alpha', 'ncons':10*r, 'J': -0.22*mV}, # note the sign
#     'II': {'synapse_dyn': 'alpha', 'ncons':10*r, 'J': -0.22*mV}, # note the sign
# }

# lscp_cfg0 = {
#     #'I': {'type': 'homogeneous', 'params': {'phi': np.pi/4, 'r': 1} },
#     'E': {'type': 'perlin', 'params': {'scale': 4, 'r':10} },
#     'I': {'type': 'perlin', 'params': {'scale': 4, 'r':10} }
# }

# conn_cfg0 = {
#     'EE': {'type':'Gaussian', 'self_link':False, 'params': {'std': 9}},
#     'EI': {'type':'Gaussian', 'self_link':False, 'params': {'std': 4.5}},# EI means E -> I
#     'IE': {'type':'Gaussian', 'self_link':False, 'params': {'std': 12}}, # IE means I -> E
#     'II': {'type':'Gaussian', 'self_link':False, 'params': {'std': 6}}, 
# }
#conn_cfg0 = {'type':'Gamma', 'self_link':True, 'params': {'theta': 3, 'k': 4} }


def get_config(name='EI_net', scalar=3):
    if name=='I_net':
        pops_cfg = {
            'I': {'gs':100//scalar, 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA},
                  'cell': {'type': 'LIF', 
                           #'thr': -55*mV, 'ref': 10*ms, 'rest': -70*mV,
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  }
        }

        conn_cfg = {
            'II': {'ncons': 1000//(scalar**2), 'self_link':False, 
                   #'profile': {'type':'Gamma', 'params': {'theta': 0.5, 'kappa': 8} },
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   #'profile': {'type':'Gaussian', 'params': {'std': 3} },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
#                   'synapse': {'type':'alpha_voltage', 'params': {'J': -0.221*mV, 'delay':1*ms, 'tau': 5*ms}},
                   #'anisotropy': {'type': 'perlin', 'params': {'r': 1, 'scale':3}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/4.}}
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   },
        }    
    elif name=='EI_net':
        # pops_cfg = {
        #     'I': {'gs':60, 'mu': 350*pA, 'sigma': 100*pA},
        #     'E': {'gs':120, 'mu': 350*pA, 'sigma': 100*pA}
        # }

        # lscp_cfg = {
        #     'I': {'type': 'perlin', 'params': {'scale': 4, 'r':1} }
        # }

        # conn_cfg = {
        #     'EE': {'type':'Gaussian', 'self_link':False, 'params': {'std': 9}},
        #     'EI': {'type':'Gaussian', 'self_link':False, 'params': {'std': 12}},# EI means E -> I
        #     'IE': {'type':'Gaussian', 'self_link':False, 'params': {'std': 4.5}}, # IE means I -> E
        #     'II': {'type':'Gaussian', 'self_link':False, 'params': {'std': 6}}, 
        # }

        # syns_cfg = { 
        #     'EE': {'synapse_dyn': 'alpha', 'ncons':720, 'J': 10*pA}, # note the sign
        #     'EI': {'synapse_dyn': 'alpha', 'ncons':180, 'J': 10*pA}, # note the sign
        #     'IE': {'synapse_dyn': 'alpha', 'ncons':720, 'J': -1.76*mV}, # note the sign
        #     'II': {'synapse_dyn': 'alpha', 'ncons':180, 'J': -1.76*mV}, # note the sign
        # }
        pops_cfg = {
            'I': {'gs':60//scalar, 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  },
            
            'E': {'gs':120//scalar, 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  }
        }

        conn_cfg = {
            'EE': {'ncons': 720//(scalar**2), 'self_link':False, 
                  'profile': {'type':'Gaussian', 'params': {'std': 9/scalar} },
                  'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms} },
                  'anisotropy': {'type': 'perlin', 'params': {'scale': 3, 'r':1}}
                  #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/4.}}
                  },
            
            'EI': {'ncons': 180//(scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 12/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/4.}}
                   },
            
            'IE': {'ncons': 720//(scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 4.5/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi}}
                   },

            'II': {'ncons': 180//(scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 6/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/2.}}
                   },
        }    
    else:
        raise
    
    return pops_cfg, conn_cfg


