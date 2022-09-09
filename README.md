# Brian's anisotropic network

This is a `Brian` implementation of anisotropic network introduced in [From space to time: Spatial inhomogeneities lead to the emergence of spatiotemporal sequences in spiking neuronal networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007432). 

The code structure is simple. We have a main class called `Simulate` which sets up the network and runs the simulation. Other files are dedicated to other tasks:

- `configs`: stores some default configuration settings for convenience
- `anisofy`: samples post-synapses in an anisotropic manner
- `equations`: prepares Brian-friendly equations for neurons and synapses according to the configuration dictionaries
- `landscape`: prepares the angular and shift ($\phi, r$) landscape
- `utils`: some useful utility functions
- `viz`: a set of functions for visualizing results

## Documentation
For further details check the documentation here: [https://arashgmn.github.io/anisonet-brian](https://arashgmn.github.io/anisonet-brian)

## Setting up the environment
You basically need five things:

- brian
- numpy
- scipy
- noise (for Perlin noise)
- matplotlib

The rest are dependencies. I recommand installing all but the last one with `conda` with the same order as above, or with `mamba` -- it's just faster. If you like, you can use `environment.yml` to create your conda environment (although depending on your machine, it might not work as this environment is tailored to my Ubuntu 22 installation.).

## How to run
After setting up your environment, simply navigate to `anisonet/` and run the following command:

```python simulate.py```

This makes the *Inhibitory Network* of the aforementioned paper, warms up the netowrk and stores the state in the `results/data/` (will be created upon first run). It also visualizes landscape, in- and out-degrees, and the realized landscape, and saves these figures in the `results` directory. Finally, it runs the network for 2.5 seconds (while saving the intermediate results), and finishes by making a cool animation of the network activity (which again, you can find in `results`). *Excitatory-Inhibitory Network*, too, can be initialized by setting `net_name="EI_net"`. However, bear in mind that the code is not fully tested for that setting.

## Local installation
If you wish, after installing all the dependencies, you can install the `anisonet` as a package (prefereably in your virtual/conda environment) so that you can access it in anywhere. To install, `cd` to the repo's folder and type the following:

```pip install -e .```

and you're good to go. (**Don't foget the period at the end of the command!**)


## Important Note
This implementation is work in progress and possibly contains bugs! Any feedback is welcome.

