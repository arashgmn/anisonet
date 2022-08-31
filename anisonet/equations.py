#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:09:32 2022

@author: arash
"""

import brian2 as b2


def get_nrn_eqs(pop_name, pops_cfg, syn_base):    
    """
    It matters how the synapse inputs are modeled. Seemingly, the one used in
    NEST and the paper are using a "current-alpha", as opposed to "voltage-" or
    "conductance-": `source code`_.
    
    .. _source code: https://github.com/nest/nest-simulator/blob/d7b3c7671c5fcd9acf211b58012db9729607a1db/models/iaf_psc_alpha.h#L48
    """
    
    tmp = '''
        mu: amp (shared)
        sigma: amp (shared)
        noise_pop = mu + sigma*sqrt(1*ms)*xi_pop: amp 
        #noise_pop = mu + sigma*sqrt(brian_dt)*xi_pop: amp # changed here      
        '''
    #tmp = tmp.replace('brian_dt', str(b2.defaultclock.dt/b2.ms)+'*ms')
    
    if syn_base!='voltage':
        eqs_str= tmp + '''dv/dt = (E-v)/tau + (noise_pop + I_syn)/C : volt (unless refractory)\n'''
        eqs_str+= '''I_syn: amp \n'''
        
    else:
        eqs_str= tmp + '''dv/dt = (E-v)/tau + noise_pop/C : volt (unless refractory)\n'''
        
    eqs_str = eqs_str.replace('_pop', '_'+pop_name)
    eqs = b2.Equations(eqs_str, 
                    C = pops_cfg[pop_name]['cell']['C'],
                    E = pops_cfg[pop_name]['cell']['rest'],
                    tau = pops_cfg[pop_name]['cell']['tau'])
    
    return eqs

def get_syn_eqs(conn_name, conn_cfg, syn_base):
    """
    It matters how the synapse inputs are modeled. Seemingly, the one used in
    NEST and the paper are using a "current-alpha", as opposed to "voltage-" or
    "conductance-": `nest repo`_.
    
    Nest normalizes its models according the the following rule (`nest docs`_):
        
        - `iaf_psc_alpha`: max current of unitary excitation/inhiition = 1 pA
        - `iaf_cond_alpha`: max conductance of unitary excitation/inhiition = 1 nS
        
    In order to get the same behaviour in brian, we need to ascribe a weight to
    our unitary events. (This weight happen to be exp(1).) In addition, the 
    parameter ``h`` is unbounded, and can be well beyond 1. 
    
    #TODO: check if that's also the case in nest.
    
    .. note:
        There's a computational difference between current/conductance-based 
        synapses from voltage-based ones in brian. The former show themselves 
        in the cell equation in terms of ``I_syn``. Thus, they need to be 
        evaluated on every single timestep. One the contrary, voltage-based 
        synapses only add a particular amount of potential to the post-synapse
        voltage, and thus need to be updated once there's an event. Therefore,
        voltage-based synapse is computationally cheaper. From the 
        implementation point of view, this means current/conductance-based 
        synapses must have the ``(clock-driven)`` flag, whereas the 
        voltage-based one ``(event-driven)``.
    

    .. _nest repo: https://github.com/nest/nest-simulator/blob/d7b3c7671c5fcd9acf211b58012db9729607a1db/models/iaf_psc_alpha.h#L48
    .. _nest docs: https://nest-simulator.readthedocs.io/en/v2.20.1/models/neurons.html#classnest_1_1iaf__psc__alpha

    """
    
    # syn_type = conn_cfg[conn_cfg.keys()[0]]['synapse']
    # if 'current' in syn_type:
    #     base = 'current'            
    # elif 'voltage' in syn_type:
    #     base = 'voltage'
    # elif 'conductance' in syn_type: 
    #     base = 'conductance'
    # else:
    #     raise
        
    tmp = '''
        dg/dt = (-g+h) / tau : 1 (clock-driven)
        dh/dt = -h / tau : 1 (clock-driven)
        '''
        
    on_pre ='''h+=exp(1)'''
    on_post = ''
    
    if syn_base=='voltage':
        # J is the post-synaptic voltage increment
        eqs_str = tmp.replace('clock-driven', 'event-driven')
        eqs_str += '''w = J*g: volt \n'''
        #eqs_str += '''J = {}*mV: volt \n'''.format(J/mV)  
        eqs_str += '''J: volt (shared)\n'''  
        on_pre += '''\nv_post += w'''
        
    elif syn_base=='current':
        # J is the post-synaptic current injection 
        eqs_str = tmp
        eqs_str += '''I_syn_post = J*g: amp (summed)\n''' 
        eqs_str += '''J : amp (shared)\n'''
        
    elif syn_base=='conductance':
        # J is the maximum conductance
        eqs_str = tmp
        eqs_str += '''I_syn_post = J*g*(v_post-E): amp \n''' 
        eqs_str += '''E : volt (shared)\n'''
        eqs_str += '''J : siemens (shared)\n'''
        #eqs_str += '''J = {}*nS: siemens \n'''.format(J/nS)  
        #E = conn_cfg[conn_name]['synapse']['params']['E']/mV
        
    eqs = b2.Equations(eqs_str, 
                    tau = conn_cfg[conn_name]['synapse']['params']['tau'])
    
    return eqs, on_pre, on_post
