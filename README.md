# Brian's anisotropic network

This is a `Brian` implementation of anisotropic network introduced in [From space to time: Spatial inhomogeneities lead to the emergence of spatiotemporal sequences in spiking neuronal networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007432). 

The code structure is simple. We have a main class called `Simulate` which sets up the network and runs the simulation. Other files are dedicated to other tasks:

- `configs`: stores some default configuration settings for convenience
- `anisofy`: samples post-synapses in an anisotropic manner
- `equations`: prepares Brian-friendly equations for neurons and synapses according to the configuration dictionaries
- `landscape`: prepares the angular and shift ($\phi, r$) landscape
- `utils`: some useful utility functions
- `viz`: a set of functions for visualizing results

For further details check the documentation in `docs/index.html` (you need to clone the repo first).

## Setting up the environment
You basically need three things:

- brian
- numpy
- scipy
- matplotlib
- noise (for Perlin noise)

The rest are dependencies. I recommand installing all but last one with `conda` (or `mamba` -- it's just faster) and the last one with `pip`. Or, if you like, you can use `environment.yml` to create your conda environment (although, depending on your machine, it might not work).

## How to run
After setting up your environment, simply navigate to `anisonet/` and run the following command:

```python simulate.py```

This makes the *Inhibitory Network* of the aforementioned paper, warms up the netowrk and stores the state in `results/data/`. It also visualizes landscape, in- and out-degrees, and the realized landscape, and saves these figures in `results` directory. Finally, it runs the network for 2.5 seconds (while saving the intermediate results), and finishes by making a cool animation of the network activity (which again, you can find in `results`). 

*Excitatory-Inhibitory Network* too can be initialized by setting `net_name="EI_net"`. However, bear in mind that the code is not fully tested.

## Important Note
This implementation is work in progress and possibly contains bugs! Any feedback is welcomed.


### Known issue
There seem to by a discrepancy between the NEST implementation and Brian, specifically in the time-scaling of the stochastic noise term.
