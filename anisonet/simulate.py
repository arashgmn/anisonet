#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main module to set up and simulate an anisotropic network. Other 
modules are auxillary utilities used here. 
"""

#TODO: define the threshold, refractory and rest as equation variables and give
# them to the equation as a namespace. It's cleaner and similar to the synapse.

import os 
import logging
osjoin = os.path.join # an alias for convenient
from copy import deepcopy

import pickle
import numpy as np
import brian2 as b2
from brian2 import profiling_summary
from scipy import sparse

import time

import anisonet.viz as viz
import anisonet.utils as utils 
import anisonet.configs as configs # default configurations
import anisonet.analyze as analyze
import anisonet.equations as eq
from anisonet.landscape import generate_landscape
from anisonet.anisofy import draw_posts

from pdb import set_trace

b2.seed(81)

_plastic_models = ['tsodyks-markram']


class Simulate(object):
    """
    High-level object for simulating an anisotropic network in Brian.
    """
    
    def __init__(self, net_name='I_net', load_connectivity=True,  scalar=1,
                 result_path=None, to_event_driven = True,):
        """
        Initializes the simulator object for the given network configuration. 
        By default, tries to load the connectivity matrix from disk, otherwise
        will generate and save it. Also, prepares the net for a warmup phase 
        that can be executed via the ``warmup`` method.
        
        Upon initialization, a name is attributed to the object which is being
        used for storing figures, connectivity matrices, and states. The name
        has the following strucutre:
            
            ``<anisotropy_type>_<profile_type>_<scaling_factor>``
            
            
        :param net_name: network configuration name, either 'IE_net' or 'I_net'
            , defaults to 'I_net'
        :type net_name: str, optional
        :param scalar: A scaling factor for downsizing the network, defaults to 1
        :type scalar: int, optional    
        """
        
        if result_path==None:
            result_path = os.getcwd()
        root = osjoin(result_path, net_name)
                
        # initialize with defaults
        confs = configs.get_config(net_name, scale=scalar)
        self.pops_cfg, self.conns_cfg, self.nonuniformity, self.lscps_cfg, self.stims_cfg = confs.get()
        #self.pops_cfg, self.conns_cfg, self.stim_cfgs = configs.get_config(net_name, scalar=scalar)
        
        # processing configs
        self.process_configs(to_event_driven) 
        #self.assess_landscape()
        #self.base = self.get_synaptic_base()
        self.has_plastic = self.check_plasticity()
        
        self.load_connectivity = load_connectivity
        
        self.name = 'dummy_name'#self.generate_name(scalar, net_name) #TODO better
        
        self.res_path = osjoin(root, self.name)#+'results')
        self.data_path = osjoin(root, self.name)#+'data')
        
        # making necessary folders
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        # warm-up settings
        self.warmup_std = 500*b2.pA
        self.warmup_dur = 500*b2.ms
        
        self.fmt = '{:0>3}'
        self.state_id = 0
        self.state_str = self.fmt.format(self.state_id)
        
        self.dt = b2.defaultclock.dt
        
        self.overlay=True
        
        
    def state_initializer(self, init_cell='rand', init_syn='rand'):
        """
        A smart function that intializes the synaptic or cellular state according 
        to the desired mode, based on the provided config file.
        
        :param config: DESCRIPTION
        :type config: TYPE
        :param mode: DESCRIPTION, defaults to 'ss'
        :type mode: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE
        """
        
        msg0 = 'I am smart but ... \n'
        
        msg_mode = '''
            I only regonize steady state (ss) and random (rand) modes for now. 
            '''
        
        msg_kernel = '''
            The kernel "{}" is not among the regonized kernels. Have a look at the
            documentation.
            '''
        
        msg_cell = '''
            I only regonize LIF neuron for now.
            '''
            
        # syapse parameters must be synapse specific. However, we may induce
        # a bias in the parameters of all synapses of a certain neuron. This 
        # bias can be "anisotropic" if it reflects the location of the neuron.
        # aniso_syns = False
        # if len(mode.split('-'))>1:
        #    aniso_syns= True
        #    mode = mode.split('-')[0]
        
        # init_cell = mode.split('-')
        # if len(init_cell)>1:
        #     init_syn = init_cell[1]
        #     init_cell = init_cell[0]
        # else:
        #     init_cell = init_cell[0]
        #     init_syn = init_cell  
            
        for pop, pop_cfg in zip(self.pops, self.pops_cfg.values()):         
            # setting up voltage
            if pop_cfg['cell']['type']=='LIF':
                minV = pop_cfg['cell']['params']['rest']
                maxV = pop_cfg['cell']['params']['thr']
            else:
                raise NotImplementedError(msg0 + msg_cell)
               
            
            if init_cell=='ss':
                self.pops[pop].v = minV
            elif init_cell=='rand':
                self.pops[pop].v = '({}+ rand()*({}))*mV'.format(minV/b2.mV, (maxV-minV)/b2.mV)
            else:
                raise NotImplementedError('Initialization method not recognized')
            
            
            # setting up noise level
            self.pops[pop].mu = pop_cfg['noise']['mu']
            self.pops[pop].sigma = pop_cfg['noise']['sigma']
            
            # setting up the stimuli
            if pop not in self.stims_cfg:
                self.pops[pop].I_stim = 0*b2.nA
            else:
                #TODO: What will happen really?
                pass
        
        
        # for id_, conn_cfg in zip(range(len(self.syns)), self.conns_cfg.values()):
        #     kernel, model = conn_cfg['synapse']['type'].split('_')
        for syn_name in self.syns.keys():
            conn_cfg = self.conns_cfg[syn_name]
            kernel, model = conn_cfg['synapse']['type'].split('_')
            
            
            if init_syn=='ss':
                if kernel=='tsodyks-markram':
                    self.syns[syn_name].u = conn_cfg['synapse']['params']['U']
                    self.syns[syn_name].x = 1
                    # if 'U' not in hetrosyn_vars:
                    #     self.syns[syn_name].U = conn_cfg['synapse']['params']['U']
                
                elif kernel in ['alpha','exp','biexp', 'const']: 
                    # self.syns[syn_name].g = 0 #default is 0 anyway
                    pass
                else:
                    raise NotImplementedError(msg0 + msg_kernel.format(kernel))
                
            elif init_syn=='rand':
                if kernel=='tsodyks-markram':
                    self.syns[syn_name].u = 'rand()'
                    self.syns[syn_name].x = 'rand()'
                    # self.syns[syn_name].U = conn_cfg['synapse']['params']['U']
                    
                    # else:
                    #     self.syns[syn_name].U = utils.get_anisotropic_U(self, 
                    #                                                     syn_name,
                    #                                                     conn_cfg['synapse']['params']['U'])
                elif kernel in ['alpha','exp','biexp']:
                    self.syns[syn_name].g = 'rand()'
                elif kernel == 'const':
                    pass
                else:
                    raise NotImplementedError(msg0 + msg_kernel.format(kernel))
            
            else:
                raise NotImplementedError('Initialization method not recognized')
                
            
            self.syns[syn_name].J = conn_cfg['synapse']['params']['J']
            
            # The following ensures that the stationary response of synapse is
            # equal to the prescribed J.
            if kernel=='tsodyks-markram':
                self.syns[syn_name].J /= conn_cfg['synapse']['params']['U']
            
            if conn_cfg['training']['type']=='STDP':
                self.syns[syn_name].w  = 0.5 
            
            
        # now we impose possible non-uniformities
        for obj_name, nonunif in self.nonuniformity.items():
            # nonuniformity imposed on the population
            if len(obj_name)==1:
                for var in self.nonuniformity[obj_name].keys():
                    self.pops[obj_name].set_states(
                        {var: np.load(osjoin(self.data_path, var+'.npy'))},
                        units=False)
                
            else:
                for var in self.nonuniformity[obj_name].keys():
                    if var!='connectivity':
                        self.syns[obj_name].set_states(
                            {var: np.load(osjoin(self.data_path, var+'.npy'))},
                            units=False)
                    
        # if syn_name in self.nonuniformity:
        #     hetrosyn_vars = self.nonuniformity[syn_name].keys()
        #     if 'connectivity' in hetrosyn_vars:
        #         hetrosyn_vars.remove('connectivity')
                    
        # if 'synaptic' in self.nonuniformity[syn_name]:
        #     # msg_het = 'No anisotropy method is set of synapses. Check configs.'
        #     # assert conn_cfg['anisotropy']['synaptic']!=None, msg_het
                
        #     if kernel=='tsodyks-markram':
        #         self.syns[syn_name].u = 'rand()'
        #         self.syns[syn_name].x = 'rand()'
        #         self.syns[syn_name].U = np.load(osjoin(self.data_path, 'Us.npy'))
            
        # else:
        #     raise NotImplementedError(msg0 + msg_mode)
        
                
    def get_synaptic_base(self):
        """
        Synaptic inputs can be defined in different ways. This method reads the
        configuration keys and determines the used approach. It is important 
        when we define the governing equations.
        """
        
        syn_type = list(self.conns_cfg.values())[0]['synapse']['type']
        
        return syn_type
        # if 'current' in syn_type:
        #     return 'current'            
        # elif 'voltage' in syn_type:
        #     return 'voltage'
        # elif 'conductance' in syn_type: 
        #     return 'conductance'
        # else:
        #     raise
    
    def current_to_voltage(self):
        # TODO: maybe it is better to move it to utils.
        """
        Transfaers the current-based synapses into an (almost) equivalent 
        voltage jump increment model. Look at ``Equation`` module for more info.
        
        Note: For the ``tsodyks_markram`` kernel, we omit the steady-state term.
        """
        
        for pathway in self.conns_cfg.keys():
            syn_cfg = self.conns_cfg[pathway]['synapse']
            src, trg = pathway
            
            kernel, model = syn_cfg['type'].split('_') # identify kernel and model
            
            if model == 'jump':
                continue # no modification needed.
                msg = """
                    Pathway {} is already configured as voltage-based. Noting
                    was converted.""".format(pathway)
                logging.info(msg)
                
            elif model =='conductance':
                msg = """
                    Pathway {} is configured as conductance-based. But only 
                    current-based models can be converted to voltage-based.
                    """.format(pathway)
                logging.critical(msg)
                raise TypeError(msg)
                
            else: #model is indeed conductance-based
                # invalid kernels
                msg = """
                    kernel type {} will cause an infinite charge transfer and 
                    is not a valid option for conversion to a voltage-based 
                    model""".format(kernel)
                
                assert kernel != 'const', msg
                assert kernel != 'tsodyks_markram', msg
                
                # compute conversion factor for valid kernels
                if kernel=='exp':
                    tau_s = syn_cfg['params']['tau']
                    factor = tau_s
                    
                elif kernel=='alpha':
                    tau_s = syn_cfg['params']['tau']
                    factor = tau_s*np.exp(1)
                
                elif kernel == 'tsodyks_markram':
                    # Here we omit the constant current 
                    tau_f = syn_cfg['params']['tau_f']
                    tau_d = syn_cfg['params']['tau_d']
                    U = syn_cfg['params']['tau_d']
                    factor = (1-U)*tau_d - U*tau_f - U*(1-U)/(1./tau_f + 1./tau_d)
                    assert factor > 0
                    
                elif kernel == 'biexp':
                    tau_r = syn_cfg['params']['tau_r'] 
                    tau_d = syn_cfg['params']['tau_d'] 
                    factor = 1 # TODO: need to find the correct conversion 
                    
                else:
                    raise
                C_m = self.pops_cfg[src]['cell']['params']['C']
                syn_cfg['params']['J']*= factor/C_m
                syn_cfg['type'] = 'const_jump'
                
                self.conns_cfg[pathway]['synapse'] = syn_cfg # update
                
            
    def generate_name(self, scalar, net_name):
        # specifies the network scale
        name = 'S%.1f_'%(scalar)
        print('Initializing network: '+net_name)
        print('\t* scaled down by a factor: {}'.format(scalar))
        
        # specifies the network type
        pop_str = ''
        for pop in self.pops_cfg.keys():
            pop_str += pop
        name += pop_str+'_'
        
        print('\t* populations: '+', '.join(pop_str))
        print('\t* pathways: ')
        for item in self.conns_cfg.items():
            pw_name, pw_cfg = item
            
            name += pw_name
            
        # I factor the GA out becasue it's shared in Gamma and Gaussian
            if pw_cfg['profile']['type']=='Gaussian':
                name += 'U' 
            elif pw_cfg['profile']['type']=='Gamma':
                name += 'M'
            elif pw_cfg['profile']['type']=='homog':
                name += 'H'
            else:
                raise
                
            # TODO: anisotropy is not a good feature for naming
            # I factor the GA out becasue it's shared in Gamma and Gaussian
            # if pw_cfg['anisotropy']['type']=='perlin':
            #     name += 'P' 
            # elif pw_cfg['anisotropy']['type']=='homogeneous':
            #     name += 'H'
            # elif pw_cfg['anisotropy']['type']=='random':
            #     name += 'R'
            # elif pw_cfg['anisotropy']['type']=='iso':
            #     name += 'I'
            # else:
            #     raise
            
            # print('\t\t{}: Connectivity profle {} - anisotropy pattern {}'.format(
            #     pw_name, pw_cfg['profile']['type'], pw_cfg['anisotropy']['type']))
            
            name+='-'
        
        print('\nThe network is abbreviated to : '+ name[:-1]+'\n')
        return name[:-1] #dropping the last seperator
    
    def process_configs(self, to_event_driven):
        """
        Every population configuration must have the following keys:
            
            * **Mandatory**: 'gs', 'cell'
            * **Optional**: 'noise'
        
        Every connection configuration must have the following keys:
            
            * **Mandatory**: `synapse`
            * **Optional**: 'training' , 'profile', 'anisotropy'
        
        The entry 'anisotropy' itself must have some keys:
            
            * **Mandatory**: `connectivity`, 'synaptic'
            
        Every stimulation configuration must have the following keys:
            
            * **Optional**: everything is optional
            
        If optional entries are not provided, the simlest possible 
        case will be assumed. i.e., 
        
            * Noise: Both mean and std will be set to zero 
            * Profile: No connectivity profile will be considered. 
                This is equivalent to a random network.
            * Anisotropy: No anisotropy will be configured. This 
                means connectivity profile will be isotropic.
            * Training: Training will be disabled. Thus, synaptic
                weights will fixed (in long-term). Short-term 
                plasticity is still possible.
        """
        
        # filling the None with appropriate values
        for pathway in self.conns_cfg.keys():
            src, trg = pathway
            config = self.conns_cfg[pathway]
            
            
            # Filling the omitted optional configs
            if 'profile' not in self.conns_cfg[pathway]:
                 self.conns_cfg[pathway]['profile'] = {'type': 'homog' }
            
            if 'training' not in self.conns_cfg[pathway]:
                self.conns_cfg[pathway]['training'] = {'type': None }
            
            # if 'anisotropy' not in self.conns_cfg[pathway]:
            #     self.conns_cfg[pathway]['anisotropy'] = {'params': {},
            #                                             'connectivity': None,
            #                                             'synaptic': None}
            
            # filling None profiles with homog
            # if config['profile']==None:
            #     self.conns_cfg[pathway]['profile'] = {'type': 'homog', 'params':{}}
            
            # # filling None anisotropy with empty parameters
            # if config['anisotropy']==None:
            #      self.conns_cfg[pathway]['anisotropy'] = {'params': {},
            #                                              'connectivity': None,
            #                                              'synaptic': None}
            
            # add None for ommitted nonuniformity methods
            # if pathway not in self.nonuniformity:
            #     self.nonuniformity[pathway] = {'connectivity': None, 
            #                                     #'synaptic': None
            #                                     }
            # else:
            #     if 'connectivity' not in self.nonuniformity[pathway]:
            #         self.nonuniformity[pathway]['connectivity'] = None
            #     if 'synaptic' not in self.nonuniformity[pathway]:
            #         self.nonuniformity[pathway]['synaptic'] = None
        
        
            # adding types to anisotropy params
            # for param, value in config['anisotropy']['params'].items():
            #     if type(value)==type({}):
            #         if 'type' not in value:
            #             config['anisotropy']['params'][param]['type'] = None
            #         if 'args' not in value:
            #             config['anisotropy']['params'][param]['args'] = None
                
                
        # adding stimulation variable
        # for pop in self.pops_cfg.keys():
        #     if 'stim' not in self.pops_cfg[pop]:
        #         self.pops_cfg[pop]['stim'] = {}
        
        
        # converting to models to jump if necessary
        if to_event_driven:
            self.current_to_voltage()
        
        # checking if current models are consistent 
        for pop in self.pops_cfg.keys():
            
            for pathway in self.conns_cfg.keys():
                src, trg = pathway
                
                if trg == pop:
                    _, model = self.conns_cfg[pathway]['synapse']['type'].split('_')
                    
                    if 'input_model' in self.pops_cfg[pop]:
                        msg = """ Synaptic input models impinging on population {} are not consistent: {} and {} are both given in the pathways configuration.
                        """.format(pop, model, self.pops_cfg[pop]['input_model'])
                        assert model == self.pops_cfg[pop]['input_model'], msg
                    else:
                        self.pops_cfg[pop]['input_model'] = model
            
            
    def setup_net(self, init_cell='ss', init_syn='rand'):
        """
        Sets up a network by the following steps:
            #. defining the populations (``setup_pops``)
            #. defining the landscape (``setup_landscape``)
            #. defining the synapses (``setup_syns``)
            #. configuring the spike monintors (``configure_monitors``)
        
        and adds them all to a Brian network object for simulation. Read their 
        description on each method.
        """
        
        b2.start_scope()
        
        print('Net setup started.')
        self.setup_pops()
        self.setup_landscape()
        self.setup_syns()
        self.state_initializer(init_cell=init_cell, init_syn=init_syn)
        
        self.configure_monitors()
        self.net = b2.Network()
        self.net.add(self.pops.values())
        self.net.add(self.syns.values())
        self.net.add(self.mons)
        print('Net set up.')
        
        # for viz
        if len(list(utils.dict_extractor('phi', self.lscps)))!=len(self.pops.keys()):
            self.overlay=False
        
    def setup_pops(self):
        """
        Each population is set up from the ``pops_cfg`` which is a nested 
        dictionary. Look at ``configs`` module for further details. Neurons are
        endowed with a ``coord`` attribute that encodes their ``(x,y)`` index
        on the grid. All neurons are driven by a Gaussian background noise.
        
        Populations are accessible via the `pops` attribute of the ``Simulate``
        object, in form of a dictionary keyed by the population's name.
        
        .. note::
            First index (columns) indicate the `x`-coordinate and second one 
            (rows) the `y`-coordinate. It differs from array convention but 
            mathes the one of images.
        
        """
        print('{} -- Setting up populations.'.format(time.ctime()))
        
        self.pops = {}
        for pop_name in self.pops_cfg.keys():
            gs = self.pops_cfg[pop_name]['gs'] # grid size
            cell_cfg = self.pops_cfg[pop_name]['cell']['params']
            noise_cfg = self.pops_cfg[pop_name]['noise']
            
            # initialize population
            eqs = eq.get_nrn_eqs(pop_name, self.pops_cfg,)
            eqs = self.get_nonunif_eqs(pop_name, eqs) # adjust equations for nonuniformities
            
            pop = b2.NeuronGroup(N = gs**2, 
                                 name = pop_name, 
                                 model = eqs, 
                                 refractory = cell_cfg['ref'], #2*b2.ms, 
                                 threshold='v > {}*mV'.format(cell_cfg['thr']/b2.mV),
                                 reset='v={}*mV'.format(cell_cfg['rest']/b2.mV),
                                 method='euler'
                                 )
            # pop.mu = noise_cfg['mu']
            # pop.sigma = noise_cfg['sigma']
            # self.state_initializer(pop, self.pops_cfg[pop_name], mode='rand')
            # pop.v = np.random.uniform(cell_cfg['rest']/b2.mV,
            #                           cell_cfg['thr']/b2.mV,
            #                           pop.N) *b2.mV

            
            # add coordinates
            pop.add_attribute('coord')
            y,x = np.indices((gs,gs))
            pop.coord = list(zip(x.ravel(),y.ravel()))
            
            # add grid size for convenience
            pop.add_attribute('gs')
            pop.gs = gs
            
            # add anisotropy parameters
            self.pops[pop_name] = pop
            del x,y, cell_cfg, noise_cfg, gs, eqs
            
            
    # def assess_landscape(self):
    #     for obj_name, nonunif in self.nonuniformity.keys():
    #         lscp = self.lscpss_cfg[obj_name]
            
    #         # connectivity anisotropy
    #         if 'connectivity' in nonunif:
    #             # all connectivity methods must have r and phi
    #             assert 
                    
                    
    #         if aniso['connectivity'] == 'shift':
    #             assert 'r' in aniso['params'], "Please provide shift radius of anisotropy for pathway: " +pathway
    #             assert 'phi' in aniso['params'], "Please provide shift angle of anisotropy for pathway: " +pathway

    #         elif aniso['connectivity'] == 'positive-rotate':
    #             assert 'phi' in aniso['params'], "Please provide rotation angle of anisotropy for pathway: " +pathway
            
    #         elif aniso['connectivity'] is ['squeeze-rotate', 'positive-squeeze-rotate']:
    #             assert 'r' in aniso['params'], "Please provide sqeezing ratio of anisotropy for pathway: " +pathway
    #             assert 'phi' in aniso['params'], "Please provide rotation angle of anisotropy for pathway: " +pathway
            
    #         elif aniso['connectivity'] == None:
    #             pass
    #         else:
    #             raise TypeError ("I don't understand the profile's anisotropy method!")

    #         # synaptic anisotropy
    #         if aniso['synaptic'] in ['cos', 'sin']:
    #             assert ('tau_f' in aniso['params']) or \
    #                    ('tau_d' in aniso['params']) or \
    #                    ('U' in aniso['params']), "Do not know which synaptic variable is to be anisotrofied in pathway "+pathway
    #             assert 'phi' in aniso['params'], "Please provide angle anisotropy for pathway: " +pathway
    #         elif aniso['synaptic'] == None:
    #             pass
    #         else:
    #             raise TypeError ("I don't understand the synaptic anisotropy method!")
            
    #         # 
            
    def setup_landscape(self):
        """
        Generates the requested landscape: A dictionary keyed by ``phi`` and 
        ``r`` that indicated the dominant angle of anisotropy and the 
        radial displacement for every neuron in form of flat arrays.
        
        Landscapes are accessible via the `lscp` attribute of the ``Simulate``
        object, in form of a dictionary keyed by the population's name.
        """
        print('{} -- Setting up ladscapes.'.format(time.ctime()))

        self.lscps = {}
        for pathway, lscp in self.lscps_cfg.items():
            src, trg = pathway
            gs = self.pops_cfg[src]['gs'] # grid size
            
            self.lscps[pathway] = {}
            for param_name, cfg in lscp.items():
                self.lscps[pathway][param_name] = generate_landscape(gs, cfg)
        
                
                
                
        # for param_name, params in self.lscps_cfg.values():
            
        #     lscp_cfg = self.conns_cfg[conn_name]['anisotropy']
            
        #     # this might be inefficient for constant values
        #     # rs, phis = generate_landscape(gs=gs, aniso=anisotropy)
        #     # self.lscps[conn_name] = {'phi' : phis, 'r' :rs}
        #     self.lscps[conn_name] = {}
        #     for param, cfg in lscp_cfg['params'].items():
        #         #set_trace()
                
        #         self.lscps[conn_name][param] = generate_landscape(gs, cfg)
        
        # del rs, phis, gs, lscp_cfg, src, trg
        
    def get_nonunif_eqs(self, obj, eqs):
        if obj in self.nonuniformity:
            for var in self.nonuniformity[obj].keys():
                if var=='connectivity':
                    pass
                else:
                    if 'tau' in var:
                        unit='second'
                    elif 'C' in var:
                        unit='farad'
                    else:
                        unit ='1'
                    #set_trace()
                    
                    eqs += b2.Equations(var+':'+unit)
        return eqs
        
    def setup_syns(self, visualize=False, init='rand'):
        """
        Sets up the synapses of all pathways (keys) in ``conns_cfg``; a nested 
        dictionary whose detail is given in ``configs`` module. To understand 
        how the postsynpases are selected, particularly their radial profile 
        and the notion of anisotropy, please check `[1]`_.
        
        The connectivity matrix is saved as a sparse array (.npz) if it is not
        saved already.
       
        Synaoses are accessible via the `syns` attribute of the ``Simulate``
        object, in form of a list indexed by the sorted order of pathways.
        
        .. _[1]: https://doi.org/10.1371/journal.pcbi.1007432

        """
        print('{} -- Setting up synapses ...'.format(time.ctime()))
        
        self.syns = {}
        for pathway in sorted(self.conns_cfg.keys()):
            src, trg = pathway
            
            eqs, on_pre, on_post, namespace = eq.get_syn_eqs(pathway, self.conns_cfg)
            eqs = self.get_nonunif_eqs(pathway, eqs) # adjust equations for nonuniformities
            
            ncons = self.conns_cfg[pathway]['ncons']
            spop = self.pops[src]
            tpop = self.pops[trg]
            
            
            syn = b2.Synapses(spop, tpop, 
                              model=eqs, 
                              on_pre=on_pre,
                              on_post=on_post,
                              namespace = namespace,
                              method='exact',
                              name = 'syn_'+pathway
                              )
            
            # load or save connectivity 
            w_name = self.name+'_w_'+pathway
            if self.load_connectivity:
                try:
                    print('\tLoading connectivity matrix: {}'.format(w_name))
                    w = sparse.load_npz(osjoin(self.data_path, w_name+'.npz'))
                    delays = np.load(osjoin(self.data_path, 'delays.npy'))
                except Exception as e: 
                    print(e)
                    print('\tWarning: Connecitivy file {} was not found.'.format(w_name))                    
                    print('\tWarning: Computing connectivity from scratch.')                    
                    self.load_connectivity = False
                    
                    
            # computing anisotropic post-synapses
            syn_params = {}
            nonunif = {}
            if pathway in self.nonuniformity:
                nonunif = self.nonuniformity[pathway]
                
            # these kws will be used for anisotropic linking
            kws = dict(s_coord = None, local_landscape = None, 
                       ncons = ncons,
                       srow= spop.gs, scol= spop.gs,
                       trow= tpop.gs, tcol= tpop.gs,
                       profile= self.conns_cfg[pathway]['profile'], 
                       nonuniformity = nonunif,
                       self_link= self.conns_cfg[pathway]['self_link'],
                       recurrent= trg==src,
                       )
            for s_idx in range(len(spop)):
                if self.load_connectivity:
                    t_idxs = w.col[w.row==s_idx]
                
                else:
                    local_lscp={}
                    if pathway in self.lscps:
                        local_lscp = utils.get_local_lscp(s_idx, self.lscps[pathway]) 
                            
                    kws.update({'s_coord': spop.coord[s_idx],
                                'local_landscape': local_lscp})
                    # adding the methods
                    # aniso_methods = deepcopy(self.conns_cfg[key]['anisotropy'])
                    # aniso_methods.pop('params')
                    # kws['aniso_methods'] = aniso_methods
                    
                    #set_trace()
                    s_coord, t_coords, syn_param = draw_posts(**kws) # projects s_coord
                    t_idxs = utils.coord2idx(t_coords, tpop)
                    
                    for k,v in syn_param.items():
                        if k in syn_params:
                            syn_params[k] = np.concatenate([syn_params[k],v])
                        else:
                            syn_params[k] = v
                    
                syn.connect(i = s_idx, j = t_idxs)
                
            # Setting up delays
            # syn.J = self.conns_cfg[key]['synapse']['params']['J']
            # syn.delay = np.array(delays).ravel()*b2.ms
            #syn.delay = np.random.uniform(0.5, 2.5, len(syn.delay))*b2.ms
            
            #self.state_initializer(syn, self.conns_cfg[key], mode='rand')
            syn.add_attribute('is_plastic')
            syn.is_plastic = False
            for plastic_model in _plastic_models:
                if plastic_model in self.conns_cfg[pathway]['synapse']['type']:
                    syn.is_plastic = True 
            
            
            # append to the class
            self.syns[pathway] = syn
            #set_trace()
            # save if not saved
            if not self.load_connectivity:
                row_idx = np.array(syn.i) # pre
                col_idx = np.array(syn.j) # post
                data = np.ones_like(row_idx)
                w = sparse.coo_matrix((data, (row_idx, col_idx)))
                sparse.save_npz(osjoin(self.data_path, w_name+'.npz'), w)
                for k,v in syn_params.items():
                    np.save(osjoin(self.data_path, k), v)
                    
                    
                del t_coords, s_coord, kws, row_idx, col_idx, data
        del src, trg, eqs, on_pre, on_post, ncons 
        del spop, tpop, syn, t_idxs, w
     
        
    def configure_monitors(self):
        """
        Defines a list of spike monitor called ``mon_<population_name>`` for 
        each population, and adds them to the ``Simulate`` object.
        
        The monitors are accessible via the `mons` attribute in form of a list 
        indexed by the sorted order of populations.
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        print('{} -- Configuring monitors'.format(time.ctime()))
        
        # this is necessary to delete mons. Otherwise it won't work
        if hasattr(self, 'mons'):
            del self.mons
            
        self.mons = []
        for pop_name in sorted(self.pops.keys()):
            # self.mons.append(b2.StateMonitor(self.pops[pop_name], 
            #                                  variables='v', 
            #                                  record=True))
            self.mons.append(b2.SpikeMonitor(self.pops[pop_name], 
                                             record=True, name='mon_'+pop_name))

        for syn in self.syns.values():
            if syn.is_plastic:    
                self.mons.append(b2.StateMonitor(syn, variables=['u','x'], 
                                                 record=True, dt = 500*b2.ms,
                                                 name='mon_'+syn.name))        
            
            #self.mons.append(b2.StateMonitor(self.syns[0], variables=['x','u','g','g_tmp'], record=True, name='syn_'+pop_name))
            #self.mons.append(b2.StateMonitor(self.pops['I'], variables=['I_syn_I'], record=True, name='pop_'+pop_name))
    
    def warmup(self):
        """
        Warms up the neurons for 500 ms. In the first half neurons receive no
        input rather than a 500 pA (std) white noise. In the second half the
        background activity (mean and std) will be adjusted to their simulation 
        values.
        
        After the each warm-up, the states is saved on the disk for restoration
        for later simulations. 
        
        
        .. note::
            Previous warm-up states will be overwritten by the freshest 
            execution.
        """
        
        # resetting the time
        self.net.t_ = 0
        # we don't need to record warm-up activities        
        # for mon in self.mons:
        #     mon.active = False
        
        # Let's save mu and sigma for later and swich them off for now
        mus ={}
        stds = {}
        for pop in self.pops.values():
            mus[pop.name] = pop.mu/b2.pA
            stds[pop.name] = pop.sigma/b2.pA
            
            pop.mu = 0*b2.pA # turning background off
            pop.sigma = self.warmup_std # warm up noise
            
        print('{} -- Starting warm-up ...'.format(time.ctime()))
        
        self.net.run(self.warmup_dur/2)
        for pop in self.pops.values():
            pop.mu = mus[pop.name]*b2.pA
            pop.sigma = stds[pop.name]*b2.pA
        self.net.run(self.warmup_dur/2)
        print('Finished warm up. Storing results.')
        
        self.net.store(name= self.name, 
                       filename= osjoin(self.data_path, self.name+'.wup'))
        
        # switch on monitors        
        # for mon in self.mons:
        #     mon.active = True
        
        del mus, stds            
    
        
    def start(self, duration=2000*b2.ms, batch_dur=1000*b2.ms, 
              restore=True, profile=False, plot_snapshots=True,
              warmup=False):
        """
        Starts a long simulation by breaking it down to several batches. After
        each ``batch_dur``, the monitors will be saved on disk, and simulation
        monitors will be reset to combat memory consumption.
        
        :param duration: total duration of simualtion, excluding warm-up phase.
            defaults to 1000*b2.ms
        :type duration: Time quantitiy, optional
        :param batch_dur: duration of simulation batches, defaults to 200*b2.ms
        :type batch_dur: Time quantitiy, optional
        :param restore: whether or not restore the warmed-up stete from disk,
            defaults to True
        :type restore: bool, optional
        :param profile: whether or not profile the simulation (useful for 
            performance analysis), defaults to False
        :type profile: bool, optional
        :param plot_snapshots: whether or not plot the firing rate at the end of 
            each batch, defaults to True
        :type plot_snapshots: bool, optional
        """
        
        # we try to restore state if requested, but throw a warning if couldn't
        if restore:
            try:
                self.net.restore(name = self.name, 
                                 filename= osjoin(self.data_path, self.name+'.wup'))
                print('Restored state from state: {}'.format(self.name+'.wup'))
            
            except Exception as e: 
                print(str(e))
                print('Warning: Could not restore state from: {}.'.format(self.name+'.wup'))
                restore = False
                
        # we proceed with the requested warmup only if nothing is restored
        if (warmup) and (not restore):
            self.warmup()
        
        print('Starting simulation.')
        nbatch = int(duration/batch_dur)
        if (duration-nbatch*batch_dur)/(self.dt)>0:
            nbatch += 1
        
        # fmt = '{:0>%d}'%(np.log10(nbatch)+1) #suffix_format
        
        for n in range(nbatch):
            self.reset_monitors()
            
            dur = min(batch_dur, duration-n*batch_dur)
            print('{} -- Starting simulation part {}/{}'.format(time.ctime(), n+1, nbatch))
            self.net.run(dur, profile=profile)
            
            if profile:
                print(profiling_summary(self.net))
            
            if plot_snapshots:  
                viz.plot_firing_rates(sim=self, suffix='_'+self.state_str)
            
            print('before: '+self.state_str)                
            self.save_monitors()
            print('after: '+self.state_str)                
                    
    
    def update_state(self):
        self.state_id += 1
        self.state_str = self.fmt.format(self.state_id)
        
        
    def save_monitors(self):
        for mon in self.mons:
            data = mon.get_states()
            path = osjoin(self.data_path, 
                          self.name+'_'+mon.name+'_'+self.state_str+'.dat')
            with open (path, 'wb') as f:
                pickle.dump(data, f)
        
        # saving monitors means the a new batch will be configured next
        # so we can update the sate safely.
        self.update_state()
        
        del mon, data, path
    
    def make_txy(self):
        import glob
        
        for pop_name in self.pops.keys():
            txy = []
            files_list = sorted(glob.glob(osjoin(self.data_path, 
                                                 self.name+'_mon_'+pop_name+'*.dat')))
            for file in files_list[-2:]:
                f = open(file, 'rb')
                f = f.read()
                data = pickle.loads(f)
                
                xy = utils.idx2coords(data['i'], self.pops[pop_name])
                txy.append(np.hstack((data['t'].reshape(-1,1), xy)))
                del data,xy
                
            txy = np.concatenate(txy)
            np.savetxt(osjoin(self.data_path, self.name+'_'+pop_name+'_txy.csv'),
                       txy, fmt=('%f, %d, %d'))
            
            del txy, files_list
            
    def reset_monitors(self):
        """
        Resets the monitors by removing them, redefining them, and adding them
        to the network again. This is necessary in Brian (look here:
        https://brian.discourse.group/t/how-to-reset-network-monitors/548)
        """
        self.net.remove(self.mons)
        self.configure_monitors()
        self.net.add(self.mons)
        
    
    def post_process(self, ss_dur=10):
        print('{} -- Starting postprocessing ...'.format(time.ctime()))
        
        logging.info('Visualizing landscape, degress, and connectivity.')
        viz.plot_landscape(self)
        viz.plot_in_out_deg(self)
        viz.plot_realized_landscape(self)
        viz.plot_connectivity(self)
        
        logging.info('Visualizing firing rate distribution and autocorrelation.')
        viz.plot_firing_rates_dist(self)
        viz.plot_autocorr(self)
        
        logging.info('Making activity animation.')
        viz.plot_animation(self, ss_dur=ss_dur) 

        logging.info('Computing synchrony order parameter.')
        #viz.plot_R(self)
        
        # short-term weights
        if self.has_plastic:
            viz.plot_relative_weights(self)
            viz.plot_relative_weights_2d(self)
        
        # Long term weights
        viz.plot_LT_weights(self)
        
        
        analyze.find_bumps(self, plot=True)
        viz.plot_manifold(self, 2)
        
        
    def get_syn_mons(self):
        syn_mons = []
        for mon in self.mons:
            if 'syn' in mon.name:
                syn_mons.append(mon)
                
        return syn_mons
            
    def get_pop_mons(self):
        syn_mons = []
        for mon in self.mons:
            if 'syn' not in mon.name:
                syn_mons.append(mon)
                
        return syn_mons
        
    def check_plasticity(self):
        has_plasticity = False
        
        for pathway_cfg in self.conns_cfg.values():
            for model in _plastic_models:
                if model in pathway_cfg['synapse']['type']:
                    has_plasticity = True
    
        return has_plasticity 
    
    def set_protocol(self):
        stim_cfgs = utils.stimulator(self, self.stims_cfg)
        for stim_id, stim_cfg in stim_cfgs.items():
            pop, id_ = stim_id.split('_')
            
            self.pops[pop].I_stim[ stim_cfg['idxs'] ] = stim_cfg['I_stim']#.values[0]
        
    def train(self):
        assert self.training==True
        
        
if __name__=='__main__':
    pass
    #b2.defaultclock.dt = 2000*b2.us
    # I_net
    # sim = Simulate('I_net_focal_stim', scalar=2., load_connectivity=0, 
    #                 to_event_driven=1, )
    
    # sim.setup_net()
    #sim.warmup()
    # sim.set_protocol()
    # sim.start(duration=2000*b2.ms, batch_dur=1000*b2.ms, 
    #            restore=False, profile=False, plot_snapshots=True)
    
    # sim.reset_monitors()
    # sim.net.run(10*b2.ms, )
    # for pop in sim.pops.values():
    #     pop.I_stim = 0*b2.pA
    # sim.net.run(3900*b2.ms, )
    # viz.plot_firing_rates(sim, suffix='_all')
    # sim.save_monitors(suffix ='all')
    
    # sim.post_process(overlay=True)
    # import matplotlib.pyplot as plt
    
    # m = sim.mons[1]
    # s = sim.syns[0]
    
    # print(s.g[:11])
    # for i in range(10):
    #     plt.plot(m.t, m.x[i], 'r', alpha=1-i/10.)
    #     plt.plot(m.t, m.u[i], 'b', alpha=1-i/10.)
    #     plt.plot(m.t, m.g[i], 'g', alpha=1-i/10.)
    # print(m.g.max(), m.g.min())
    
    # plt.plot([],[], 'r', label='x',)
    # plt.plot([],[], 'b', label='u',)
    # plt.plot([],[], 'g', label='g',)
    # plt.legend()
    # EI_net
    # sim = Simulate('EI_net', scalar=2.5, load_connectivity=False,
    #                 voltage_base_syn=1)
    # sim.setup_net()
    # sim.warmup()
    # sim.start(duration=2000*b2.ms, batch_dur=1000*b2.ms, 
    #           restore=False, profile=False)
    # sim.post_process()
    
    # import matplotlib.pyplot as plt
    
    # ms = sim.mons[1]; mp = sim.mons[2]
    # plt.plot(mp.t[:100], mp.I_syn_I[0,:100]/mp.I_syn_I.max()/30)
    # for i in np.where(sim.syns[0].j==0)[0]:
    #     plt.plot(ms.t[:100], ms.g[i,:100])
